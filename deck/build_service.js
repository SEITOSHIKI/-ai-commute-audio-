// カンツギ「サービス説明資料」：電気工事会社の現場の方（職長・施工管理者）と経営の方（工事部長・社長・DX担当）向け。
// 16枚版に載せきれなかった、実利・用途・使う場面を説明する。見た目は tora16_theme.js、数字は tora_v2.js から。
// Usage: node build_service.js [outDir]   (needs pptxgenjs and sharp)
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const V = require("./tora_v2.js");
const TH = require("./tora16_theme.js");
const { K, X0, W, ACC, txt, box, hline, arrow, run, heading, table, NAVY, YELLOW, PALEY, PALE, GREEN, RED, GRAY7 } = TH;
const OUT_DIR = process.argv[2] || __dirname;
const OUT = path.join(OUT_DIR, "カンツギ_サービス説明資料.pptx");
const ASSETS = path.join(__dirname, "assets");

const CV = V.customerValue(), be = Math.round(CV.breakEvenMin);
const man = (v) => Math.round(v).toLocaleString("en-US");
const first = CV.cost + V.PLAN.setup;
const d200 = V.customerValue({ plan: { workDays: 200 } });
const note = (s, t) => txt(s, t, { x: X0, y: 4.98, w: W, h: 0.24, size: 7.5, color: K.gray500 });
const darkBox = (s, x, y, w, h, parts, size = 9.5) => { box(s, { x, y, w, h, fill: K.ink }); txt(s, parts, { x: x + 0.14, y, w: w - 0.28, h, size, color: K.paper, valign: "middle", psa: 2 }); };
const lab = (t, dark) => run(t, { bold: true, color: dark ? YELLOW : ACC });

// 台本（1分約300字。説明会や商談で話す想定）
const NOTE_LIST = [null,
  ["表紙", "カンツギのサービス説明です。職長がAIグラスを掛けて作業するだけで、写真台帳と日報ができ、熟練者の判断が若手の教材になります。今日は、現場の方と経営の方、それぞれの使い方をお話しします。"],
  ["困りごと", "困りごとは立場ごとに違います。職長は撮影と日報で手が止まる。施工管理者は事務所に戻ってから写真の整理と台帳。経営は、残業の上限とベテランの退職です。どれも、人を増やしても減らない時間です。"],
  ["変わること", "変わるのは、撮る・書く・探すの3つです。要所で「ここ撮って」と言えば、写真と寸法が残ります。日報と写真台帳は下書きされて届くので、確認して直すだけです。やることは、掛ける・話す・確認する、の3つだけです。"],
  ["1日の流れ", "1日の流れで見ると、朝礼で今日の手順と注意点を確認し、作業中は掛けているだけで記録が残ります。要所では図面との違いを知らせ、熟練者は離れた所から判断できます。終業時には日報と台帳の下書きができていて、翌朝、若手は昨日の判断を3分の音声で聞けます。"],
  ["職長", "職長の方がやることは、掛けることと、要所で声を出すことだけです。黒板を書いたり、撮影のために手を止めたりする必要はありません。図面と違う所があれば、その場で知らせます。"],
  ["施工管理者", "施工管理者の方には、写真台帳と日報が下書きされて届きます。写真は手順ごとに割り当てられ、黒板の項目は会話と図面から埋まります。書式は、電力会社の指定様式や官公庁の電子納品に合わせます。"],
  ["熟練者と若手", "熟練者の判断は、なぜそうするかまで含めて記録します。それを手順ごとの解説動画と、通勤中に聞ける3分の音声番組にして、若手に渡します。教材は、施工管理の経験者が確認してから配ります。"],
  ["現場の方へ", "現場の方に最初に約束することが4つあります。撮るのは作業エリアと要所だけ。本人の同意のもとで記録し、削除にも応じる。記録は評価のためではなく、技術を残すため。そして、入力は増やしません。"],
  ["経営の方へ", `経営の方には数字でお話しします。1拠点10台で、書類と写真整理の時間だけで年約${man(Math.round(CV.docs / 10) * 10)}万円。お客様の支払いはリースを含めて約${man(CV.cost)}万円なので、約${CV.ratio.toFixed(1)}倍です。お客様が元を取るには、1台1日${be}分減れば足ります。`],
  ["導入の進め方", `まず1拠点・2〜3班で3か月試します。導入前の2週間と導入後の4週間を比べ、1台1日${be}分以上減っていれば本契約、届かなければ機能を絞るか終了します。当社の導入コンサルタントが現場に入ります。`],
  ["よくある質問", "よくある質問です。現場の方からは、重さ、見られている感じ、操作、電池。経営の方からは、価値、今のアプリとの二重、情報の扱い、解約について、よく聞かれます。"],
  ["まずは1拠点から", "まずは1拠点で、3か月、数字で確かめてください。現場の勘を次の担い手へ。ありがとうございました。"],
];
const NOTES = {};
NOTE_LIST.forEach((p, n) => { if (p) NOTES[n] = `要約：${p[0]}\n\n${p[1]}`; });

async function main() {
  const pres = new pptxgen();
  const add0 = await TH.setup(pres, "カンツギ サービス説明資料", NOTES);
  const CHAPTER = [[2, "困りごと"], [3, "変わること"], [4, "使う場面"], [8, "現場の方へ"], [9, "経営の方へ"], [10, "導入"]];
  const add = (n) => {
    const s = add0(n);
    const c = n > 1 && CHAPTER.filter(([p]) => p <= n).pop();
    if (c) txt(s, c[1], { x: 6.62, y: 5.205, w: 1.28, h: 0.29, size: 7.5, color: NAVY, bold: true, align: "center", valign: "middle" });
    return s;
  };

  // ---------- 1 表紙 ----------
  {
    const s = add(1);
    s.addImage({ data: await TH.png("kantsugi-logotype-color"), x: 0.45, y: 0.38, w: 1.9, h: 1.9 * 120 / 560 });
    txt(s, "サービス説明資料（電気工事会社の現場の方・経営の方へ）", { x: 0.45, y: 0.9, w: 5.4, h: 0.26, size: 10, color: GRAY7 });
    txt(s, [run("掛けて働くだけで、", { breakLine: true }), run("写真台帳と日報ができる")], { x: 0.45, y: 1.18, w: 5.3, h: 1.0, size: 24, bold: true, color: NAVY, valign: "middle", lsm: 1.1 });
    txt(s, "AIグラスで作業を記録し、書類づくりと技能の引き継ぎを自動化します。職長・施工管理者・熟練者・若手・経営、それぞれの使い方をまとめました", { x: 0.45, y: 2.25, w: 5.3, h: 0.62, size: 10.5 });
    hline(s, 0.45, 3.0, 5.3, NAVY, 1.25);
    [["1台1日", "35分", "減る想定（職長15分＋施工管理者20分）"], ["1拠点10台で", `年約${man(Math.round(CV.docs / 10) * 10)}万円`, "書類と写真整理の時間だけ"], ["お客様が元を取るのは", `1台1日${be}分`, "試験導入の合格ライン"]].forEach(([a, b, c], i) => {
      const x = 0.45 + i * 1.8;
      box(s, { x, y: 3.15, w: 1.7, h: 1.25, fill: i === 2 ? K.ink : PALE });
      const col = i === 2 ? K.paper : K.ink;
      txt(s, a, { x: x + 0.1, y: 3.2, w: 1.5, h: 0.26, size: 8.5, color: i === 2 ? YELLOW : GRAY7, bold: true });
      txt(s, b, { x: x + 0.1, y: 3.46, w: 1.5, h: 0.45, size: 15, bold: true, color: i === 2 ? K.paper : ACC, valign: "middle" });
      txt(s, c, { x: x + 0.1, y: 3.92, w: 1.5, h: 0.44, size: 7.5, color: col });
    });
    const iw = 3.6, ih = iw / 2.382;
    s.addImage({ path: path.join(ASSETS, "shot-glass.png"), x: 6.0, y: 1.0, w: iw, h: ih });
    txt(s, "AIグラスに映る表示（架空の工事のデモ）", { x: 6.0, y: 1.0 + ih + 0.03, w: iw, h: 0.22, size: 8, color: GRAY7 });
    const mw = 3.6, mh = mw / 1.647;
    s.addImage({ path: path.join(ASSETS, "shot-manager.png"), x: 6.0, y: 2.85, w: mw, h: mh > 2.2 ? 2.2 : mh, sizing: { type: "cover", w: mw, h: 2.0 } });
    txt(s, "事務所に届く進み具合と報告書の下書き", { x: 6.0, y: 4.88, w: mw, h: 0.22, size: 8, color: GRAY7 });
    box(s, { x: 0.45, y: 4.55, w: 5.3, h: 0.5, fill: PALEY });
    txt(s, "この資料の数字は事業計画の試算です。値段は見積、削減時間は仮説で、試験導入で測ります", { x: 0.55, y: 4.55, w: 5.1, h: 0.5, size: 8.5, valign: "middle" });
    box(s, { x: 6.6, y: 5.2, w: 3.28, h: 0.305, line: NAVY });
    txt(s, [run("件名 ", { color: GRAY7 }), run("カンツギ サービス説明資料", { bold: true, color: NAVY })], { x: 6.68, y: 5.2, w: 2.6, h: 0.305, size: 8, valign: "middle" });
    txt(s, "1／12", { x: 9.3, y: 5.2, w: 0.58, h: 0.305, size: 9, bold: true, color: NAVY, align: "center", valign: "middle" });
    s.addShape("line", { x: 9.3, y: 5.2, w: 0, h: 0.305, line: { color: NAVY, width: 0.75 } });
  }

  // ---------- 2 困りごと ----------
  {
    const s = add(2);
    heading(s, "困りごと：書類と引き継ぎが、現場の時間を取っている", "立場ごとに困っていることが違う。どれも人を増やしても減らない時間");
    const cols = [
      ["職長", ["作業の手を止めて、撮影と黒板の書き込み", "日報は終業後に手で書く", "図面を見に車や事務所へ戻る", "若手に同じことを何度も説明する"], "日報・撮影 1日約15分（想定）"],
      ["施工管理者", ["事務所に戻ってから写真の整理と写真台帳", "どの写真がどの手順か分からない", "撮り忘れに後から気づき、撮り直しになる", "現場に行かないと進み具合が分からない"], "写真整理・台帳 1日約20分（想定）"],
      ["工事部長・経営", ["2024年から残業は原則 月45時間まで", "55歳以上が36.7%。ベテランの判断が退職で消える", "若手の独り立ちまでに年数がかかる", "手戻りの原因が、後から追えない"], "人を増やしても書類の時間は減らない"],
    ];
    const cw = 2.95, gap = 0.175, top = 1.15;
    cols.forEach(([role, items, num], i) => {
      const x = X0 + i * (cw + gap);
      box(s, { x, y: top, w: cw, h: 0.42, fill: K.ink });
      txt(s, role, { x: x + 0.14, y: top, w: cw - 0.28, h: 0.42, size: 12, bold: true, color: K.paper, valign: "middle" });
      box(s, { x, y: top + 0.42, w: cw, h: 1.85, fill: PALE });
      txt(s, items.map((t, k) => run("・" + t, { breakLine: k < items.length - 1 })), { x: x + 0.14, y: top + 0.5, w: cw - 0.28, h: 1.72, size: 10, psa: 5 });
      box(s, { x, y: top + 2.35, w: cw, h: 0.5, line: ACC, lw: 1.25 });
      txt(s, num, { x: x + 0.12, y: top + 2.35, w: cw - 0.24, h: 0.5, size: 9.5, bold: true, color: ACC, valign: "middle" });
    });
    darkBox(s, X0, 4.2, W, 0.55, [lab("AIグラスで変えられるのは3つ：", true), run("記録する（撮る・書く）・整理する（写真台帳・日報）・引き継ぐ（熟練者の判断を教材に）")], 10);
    note(s, "年齢の比率は国土交通省（総務省「労働力調査」2024年平均）。削減時間は仮説で、試験導入で測る");
  }

  // ---------- 3 使う前と使った後 ----------
  {
    const s = add(3);
    heading(s, "変わること：撮る・書く・探すを、AIが代わりにやる", "やることは「掛ける・話す・確認する」の3つだけ。入力の手間は増やさない");
    table(s, [["仕事", "いま", "使った後", "誰の時間"],
      ["撮影と黒板", "作業を止めて撮る。黒板を書く", "要所で「ここ撮って」と言うだけ。黒板の項目は会話と図面から埋まる", "職長"],
      ["日報", "終業後に手で書く", "会話と記録から下書きされる。確認して直すだけ", "職長"],
      ["写真の整理・写真台帳", "事務所で写真を選び、台帳に貼る", "手順ごとに写真を割り当てた台帳の下書きが届く", "施工管理者"],
      ["図面との照合", "現場で測って、図面と見比べる", "図面との違いをその場で知らせる（表示付きのグラス）", "職長・若手"],
      ["熟練者の判断", "口頭で伝わり、記録に残らない", "「なぜそうするか」を動画と音声ごと記録する", "熟練者"],
      ["若手への説明", "同じことを何度も説明する", "手順ごとの解説動画と、3分の音声番組で渡す", "若手"]],
      { x: X0, y: 1.15, w: W, colW: [1.75, 2.45, 3.85, 1.15], rowH: 0.42, size: 9.5, leftCols: [1, 2], name: "beforeafter" });
    box(s, { x: X0, y: 4.25, w: W, h: 0.62, fill: PALEY });
    txt(s, [lab("録画は端末に残す："), run("AIに送るのは大事な場面だけ（解析する場面に月の上限）。だから通信量と費用が読める。休憩・移動・私語は記録しない")], { x: X0 + 0.14, y: 4.25, w: W - 0.28, h: 0.62, size: 9.5, valign: "middle" });
  }

  // ---------- 4 1日の流れ ----------
  {
    const s = add(4);
    heading(s, "使う場面：1日の流れで見ると", "職長がAIグラスを掛けるのは、朝礼から終業まで。翌朝、若手が前日の判断を聞く");
    const steps = [
      ["8:00", "朝礼", "今日の手順と注意点を確認。過去にミスがあった所を共有"],
      ["8:30", "作業開始", "掛けるだけで記録が始まる。撮るのは作業エリアだけ"],
      ["10:40", "要所", "「ここ撮って」で写真と寸法を記録。図面と違えばその場で知らせる"],
      ["13:00", "遠隔の確認", "熟練者が離れた所から映像を見て判断。その判断も記録"],
      ["16:30", "終業", "日報と写真台帳の下書きができている。確認して承認"],
      ["翌朝", "振り返り", "若手が昨日の勘どころを3分の音声で聞く"],
    ];
    const cw = 1.45, gap = 0.1, top = 1.25;
    hline(s, X0 + 0.2, top + 0.24, W - 0.4, K.rule, 2);
    steps.forEach(([t, h, d], i) => {
      const x = X0 + i * (cw + gap);
      box(s, { x: x + 0.3, y: top, w: cw - 0.6, h: 0.48, fill: i === 5 ? ACC : K.ink });
      txt(s, t, { x: x + 0.3, y: top, w: cw - 0.6, h: 0.48, size: 11, bold: true, color: K.paper, align: "center", valign: "middle" });
      txt(s, h, { x, y: top + 0.6, w: cw, h: 0.3, size: 11, bold: true, color: NAVY, align: "center" });
      box(s, { x, y: top + 0.95, w: cw, h: 1.45, fill: PALE });
      txt(s, d, { x: x + 0.08, y: top + 1.0, w: cw - 0.16, h: 1.35, size: 9, valign: "top" });
    });
    table(s, [["", "職長", "施工管理者", "熟練者", "若手"],
      ["この1日で減る・残るもの", "撮影・黒板・日報 約15分", "写真整理・台帳 約20分", "判断が記録に残る", "前日の勘どころを聞ける"]],
      { x: X0, y: 3.9, w: W, colW: [2.0, 1.8, 1.8, 1.8, 1.8], rowH: [0.28, 0.38], size: 9, align: "center", leftCols: [0], name: "day" });
    note(s, "時刻は例。削減時間は1台1日の想定（仮説）で、試験導入で測る");
  }

  // ---------- 5 職長 ----------
  {
    const s = add(5);
    heading(s, "職長：要所を声で残すだけで、記録が終わる", "ケーブル接続の例。手を止めずに、寸法と判断が残る（デモは架空の工事）");
    const lw = 5.3, lh = lw / 2.382;
    s.addImage({ path: path.join(ASSETS, "shot-glass.png"), x: X0, y: 1.15, w: lw, h: lh });
    [["見つける", "図面は120mm、実物は112mm。8mm短いことをその場で知らせる"], ["直す", "熟練者の「テープ基準で測り直す」という指示も記録"], ["まとめる", "写真は手順ごとに報告書へ。報告書に使う時間は0分"]].forEach(([k, v], i) => {
      const y = 1.15 + lh + 0.12 + i * 0.36;
      box(s, { x: X0, y: y + 0.03, w: 0.8, h: 0.28, fill: K.ink });
      txt(s, k, { x: X0, y: y + 0.03, w: 0.8, h: 0.28, size: 9, bold: true, color: K.paper, align: "center", valign: "middle" });
      txt(s, v, { x: X0 + 0.9, y, w: lw - 0.9, h: 0.34, size: 9.5, valign: "middle" });
    });
    const rx = 6.0, rw = 3.6;
    box(s, { x: rx, y: 1.15, w: rw, h: 1.45, fill: PALE });
    txt(s, [lab("やること", false), run("", { breakLine: true }), run("① グラスを掛ける", { breakLine: true }), run("② 要所で「ここ撮って」と言う", { breakLine: true }), run("③ 終業時に日報の下書きを確認する")], { x: rx + 0.14, y: 1.15, w: rw - 0.28, h: 1.45, size: 10, valign: "middle", psa: 3 });
    box(s, { x: rx, y: 2.72, w: rw, h: 1.3, line: ACC, lw: 1.25 });
    txt(s, [lab("やらなくていいこと", false), run("", { breakLine: true }), run("・撮影のために手を止める", { breakLine: true }), run("・黒板を書く", { breakLine: true }), run("・日報を一から書く")], { x: rx + 0.14, y: 2.72, w: rw - 0.28, h: 1.3, size: 10, valign: "middle", psa: 3 });
    darkBox(s, rx, 4.12, rw, 0.78, [lab("グラス：", true), run("約50gのスポーツ型（記録用）。図面を手元に出す班は表示付き。粉じん・高所の現場は産業用も選べる")], 8.5);
  }

  // ---------- 6 施工管理者 ----------
  {
    const s = add(6);
    heading(s, "施工管理者：写真台帳と日報が、下書きされて届く", "AIが手順ごとに写真を割り当て、黒板の項目を会話と図面から埋める。確認して承認するだけ");
    const ph = 3.75, pw = ph * 0.707;
    s.addImage({ path: path.join(ASSETS, "sample-phototable-page.png"), x: X0, y: 1.15, w: pw, h: ph });
    txt(s, "写真台帳の見本（架空の工事）", { x: X0, y: 1.15 + ph + 0.02, w: pw, h: 0.2, size: 7.5, color: GRAY7 });
    const rx = X0 + pw + 0.3, rw = W - pw - 0.3, mh = 2.22;
    s.addImage({ path: path.join(ASSETS, "shot-manager.png"), x: rx, y: 1.15, w: rw, h: rw / 1.647, sizing: { type: "crop", x: 0, y: 0, w: rw, h: mh } });
    txt(s, "事務所の画面：進み具合と、手順ごとに割り当てた写真（デモ）", { x: rx, y: 1.15 + mh + 0.02, w: rw, h: 0.2, size: 7.5, color: GRAY7 });
    const items = [["書式は発注者ごとに合わせる", "電力会社の指定様式の写真台帳、官公庁の電子納品（デジタル写真管理情報基準）、自社の施工管理アプリへの出力"],
      ["撮り忘れがその日のうちに分かる", "手順ごとに写真がそろっているかを見せる。翌日の撮り直しを減らす"],
      ["現場に行かなくても進み具合が分かる", "会話と映像から、どの手順まで終わったかを判定する"]];
    const iw = (rw - 0.2) / 3;
    items.forEach(([h, d], i) => {
      const x = rx + i * (iw + 0.1), y = 1.15 + mh + 0.3;
      box(s, { x, y, w: iw, h: 1.22, fill: PALE });
      txt(s, [run(h, { bold: true, color: NAVY, breakLine: true }), run(d, { fontSize: 8.5, color: K.ink })], { x: x + 0.1, y: y + 0.08, w: iw - 0.2, h: 1.1, size: 9.5, valign: "top", psa: 3 });
    });
  }

  // ---------- 7 熟練者と若手 ----------
  {
    const s = add(7);
    heading(s, "熟練者と若手：判断を残し、教材で渡す", "ベテランの「なぜそうするか」を記録し、手順ごとの動画と3分の音声にする");
    const iw = 6.0, ih = iw / 2.273;
    s.addImage({ path: path.join(ASSETS, "shot-training.png"), x: X0, y: 1.15, w: iw, h: ih });
    txt(s, "手順ごとの解説動画（左）と、通勤中に聞く音声番組の台本（右）。デモ", { x: X0, y: 1.15 + ih + 0.02, w: iw, h: 0.2, size: 7.5, color: GRAY7 });
    const rx = X0 + iw + 0.25, rw = W - iw - 0.25;
    box(s, { x: rx, y: 1.15, w: rw, h: 1.3, fill: K.ink });
    txt(s, [lab("熟練者", true), run("", { breakLine: true }), run("離れた所から映像を見て「測り直せ」と指示。その判断が理由ごと記録に残る。誰の技術かを名前で残す（本人の同意のうえ）")], { x: rx + 0.12, y: 1.15, w: rw - 0.24, h: 1.3, size: 9, color: K.paper, valign: "middle", psa: 2 });
    box(s, { x: rx, y: 2.55, w: rw, h: 1.2, fill: PALE });
    txt(s, [lab("若手", false), run("", { breakLine: true }), run("手順ごとの解説動画で予習し、前日の勘どころを3分の音声で聞く。独り立ちまでの月数で効果を測る")], { x: rx + 0.12, y: 2.55, w: rw - 0.24, h: 1.2, size: 9, valign: "middle", psa: 2 });
    darkBox(s, X0, 4.2, W, 0.66, [lab("教材の確認：", true), run("当社の現場監修（施工管理の経験者）が「勘どころ」を確かめてから配る。教材の一部は業界団体や学校で使えるよう無償で公開する")], 9.5);
  }

  // ---------- 8 現場の方へ ----------
  {
    const s = add(8);
    heading(s, "現場の方へ：監視ではなく、技術を残す道具", "使う人が嫌がる仕組みは続かない。最初に4つを約束する");
    const promises = [["撮るのは作業エリアと要所だけ", "休憩・移動・私語は記録しない"], ["本人の同意のもとで記録する", "見たい・消したいの依頼にも応じる"], ["評価のためではなく、技術を残すため", "誰の技術かを名前で残す（同意のうえ）"], ["入力を増やさない", "掛ける・話す・確認する、だけ"]];
    const cw = 2.85, ch = 1.2, top = 1.15;
    promises.forEach(([h, d], i) => {
      const x = X0 + (i % 2) * (cw + 0.15), y = top + Math.floor(i / 2) * (ch + 0.15);
      box(s, { x, y, w: 0.42, h: ch, fill: i === 3 ? ACC : K.ink });
      txt(s, String(i + 1), { x, y, w: 0.42, h: ch, size: 16, bold: true, color: K.paper, align: "center", valign: "middle" });
      box(s, { x: x + 0.42, y, w: cw - 0.42, h: ch, fill: PALE });
      txt(s, [run(h, { bold: true, color: NAVY, breakLine: true }), run(d, { fontSize: 9, color: GRAY7 })], { x: x + 0.54, y, w: cw - 0.66, h: ch, size: 10.5, valign: "middle", psa: 4 });
    });
    const rx = X0 + 2 * cw + 0.35, rw = W - 2 * cw - 0.35;
    txt(s, "データの扱い", { x: rx, y: top, w: rw, h: 0.26, size: 11, bold: true, color: NAVY });
    table(s, [["項目", "扱い"], ["保存", "会社ごとに分けて当社の基盤に保存（3年）"], ["見る人", "権限で決める（班・現場・事務所）"], ["図面", "会社ごとに分け、外に出さない"], ["端末", "管理ツール（MDM）で管理。紛失時は遠隔で止める"], ["削除", "本人と会社の依頼で消す"]],
      { x: rx, y: top + 0.3, w: rw, colW: [0.75, rw - 0.75], rowH: 0.36, size: 8.5, leftCols: [1], name: "data" });
    box(s, { x: X0, y: 4.0, w: W, h: 0.86, fill: PALEY });
    txt(s, [lab("試験導入の前に確かめること："), run("保護めがねとの併用・技適・国内の正規販売・電池の持ち（要所だけの記録で1日もつか）。厳しい現場は産業用の端末を選べる", {})], { x: X0 + 0.14, y: 4.0, w: W - 0.28, h: 0.86, size: 9.5, valign: "middle" });
  }

  // ---------- 9 経営の方へ ----------
  {
    const s = add(9);
    heading(s, `経営の方へ：1拠点10台で年約${man(Math.round(CV.docs / 10) * 10)}万円。支払いの約${CV.ratio.toFixed(1)}倍`, "効果はAIグラスを掛けた班の分だけ数える。書類と写真整理の時間だけで、手戻りの削減は入れない");
    const lw = 5.3, top = 1.15;
    box(s, { x: X0, y: top, w: lw, h: 1.05, fill: PALE });
    txt(s, [lab("式："), run(`10台 × 年240日 ×（職長15分×約${man(Math.round(V.H.electrician * 1e2) * 100)}円 ＋ 施工管理者20分×約${man(Math.round(V.H.manager * 1e2) * 100)}円）`, { breakLine: true }), run(`＝ 年 約${man(CV.docs)}万円`, { bold: true, fontSize: 18, color: ACC })], { x: X0 + 0.14, y: top, w: lw - 0.28, h: 1.05, size: 9.5, valign: "middle", psa: 3 });
    txt(s, `時給＝統計の年収×会社負担1.17÷残業を含む年1,938時間。浮く時間は年${man(CV.freedHours)}時間（10台）`, { x: X0, y: top + 1.1, w: lw, h: 0.3, size: 8.5, color: GRAY7 });
    const bars = [["効果（書類の時間）", CV.docs, ACC], ["お客様の支払い", CV.cost, K.ink]];
    const sc = (lw - 2.9) / CV.docs;
    bars.forEach(([n, v, c], i) => {
      const y = top + 1.42 + i * 0.4;
      txt(s, n, { x: X0, y, w: 1.85, h: 0.36, size: 9.5, bold: true, valign: "middle" });
      box(s, { x: X0 + 1.9, y: y + 0.06, w: v * sc, h: 0.24, fill: c });
      txt(s, man(v) + "万円", { x: X0 + 1.95 + v * sc, y, w: 0.9, h: 0.36, size: 10, bold: true, valign: "middle" });
    });
    table(s, [["使う日数", "効果", "÷支払い", "元が取れる分"], ["240日", `${man(CV.docs)}万円`, CV.ratio.toFixed(2) + "倍", `${be}分`], ["200日", `${man(d200.docs)}万円`, d200.ratio.toFixed(2) + "倍", `${Math.round(d200.breakEvenMin)}分`]],
      { x: X0, y: top + 2.25, w: lw, colW: [1.2, 1.4, 1.2, 1.5], rowH: 0.26, size: 9, align: "center", name: "days" });
    const rx = 6.0, rw = 3.6;
    table(s, [["1拠点10台の支払い", "万円"], ["当社の利用料（基盤＋10台）", "300"], ["AIグラスのリース（お客様名義）", man(V.leaseSite())], [{ text: "2年目からの合計（毎年）", bold: true }, { text: man(CV.cost), bold: true }], ["1年目は＋導入支援", "200"], [{ text: "1年目の合計", bold: true }, { text: man(first), bold: true }]],
      { x: rx, y: top, w: rw, colW: [2.6, 1.0], rowH: 0.3, size: 9, strongRows: [3, 5], name: "pay" });
    box(s, { x: rx, y: top + 1.95, w: rw, h: 1.15, fill: K.ink });
    txt(s, [lab("お客様が元を取る条件", true), run("", { breakLine: true }), run(`1台1日 ${be}分`, { bold: true, fontSize: 20, breakLine: true }), run("試験導入で測り、届かなければ本契約しない", { fontSize: 9 })], { x: rx + 0.14, y: top + 1.95, w: rw - 0.28, h: 1.15, size: 10, color: K.paper, valign: "middle", psa: 2 });
    box(s, { x: X0, y: 4.35, w: W, h: 0.52, line: ACC, lw: 1.25 });
    txt(s, [lab("自社の数字で：", false), run("導入効果シミュレーター（台数・削減時間・年収・使う日数を入れると、効果と倍率と元が取れる分が出る）で、一緒に計算できます")], { x: X0 + 0.14, y: 4.35, w: W - 0.28, h: 0.52, size: 9.5, valign: "middle" });
  }

  // ---------- 10 導入の進め方 ----------
  {
    const s = add(10);
    heading(s, "導入の進め方：まず1拠点・2〜3班で3か月", "導入前の2週間と導入後の4週間を比べ、数字で決める");
    const ph = [["1か月目", "準備", "現状調査・端末の設定・図面と手順の登録・報告書の型の登録・研修（2回）"], ["2週間", "いまを測る", "導入前の日報・写真整理・台帳の時間を、職長と施工管理者が記録"], ["4週間", "使う", "職長がグラスを掛けて使う。当社の導入コンサルタントが週2日立ち会う"], ["判定", "決める", `1台1日${be}分以上で本契約。15〜${be}分は機能を絞って続ける。15分未満は終了`]];
    const cw = 2.2, gap = 0.133, top = 1.15;
    ph.forEach(([t, h, d], i) => {
      const x = X0 + i * (cw + gap);
      box(s, { x, y: top, w: cw, h: 0.5, fill: i === 3 ? ACC : K.ink });
      txt(s, [run(t + "　", { fontSize: 9 }), run(h, { bold: true, fontSize: 12 })], { x: x + 0.12, y: top, w: cw - 0.24, h: 0.5, color: K.paper, valign: "middle" });
      box(s, { x, y: top + 0.5, w: cw, h: 1.35, fill: PALE });
      txt(s, d, { x: x + 0.1, y: top + 0.56, w: cw - 0.2, h: 1.25, size: 9.5 });
      if (i < 3) arrow(s, x + cw + 0.01, top + 0.25, gap - 0.02, 0, { lw: 2 });
    });
    table(s, [["", "当社", "お客様"],
      ["人", "導入コンサルタント（約2か月・42人日）、現場監修（施工管理の経験者）", "DX担当1名、職長2〜3名、施工管理者1〜2名"],
      ["決めること", "撮影の計画（何を・誰が・何時間）、報告書の型、測り方", "対象の現場と班、合格ラインの確認"],
      ["費用", "導入支援 200万円（着手金50%、結果で本契約に切り替え）", "試験導入中のグラスは当社がレンタルして渡す"]],
      { x: X0, y: 3.2, w: W, colW: [1.1, 4.4, 3.7], rowH: [0.28, 0.42, 0.42, 0.42], size: 9, leftCols: [1, 2], name: "roles" });
  }

  // ---------- 11 よくある質問 ----------
  {
    const s = add(11);
    heading(s, "よくある質問：現場の方から・経営の方から", "答えは数字と約束で。分からないことは、試験導入で測ると答える");
    const field = [["重くないか、邪魔にならないか", "約50gで普段のメガネに近い。粉じん・高所は産業用も選べる"], ["ずっと見られているのでは", "撮るのは作業エリアと要所だけ。本人の同意のもとで、削除もできる"], ["操作を覚えるのが大変では", "掛ける・話す・確認するだけ。初日に研修を2回行う"], ["電池はもつのか", "連続撮影は短いので要所だけ記録する。1日もつかを試験導入で確かめる"]];
    const mgmt = [["本当に年300万円の価値があるか", `お客様が元を取るのは1台1日${be}分。測ってから決められる`], ["今の施工管理アプリと二重にならないか", "写真と台帳をアプリへ出力する（個社設定）"], ["図面や映像が外に出ないか", "会社ごとに分けて保存し、権限と削除を決める。端末は管理ツールで管理"], ["途中でやめられるか", "年額の契約。更新の3か月前に効果を一緒に振り返る"]];
    const col = (x, title, rows, dark) => {
      txt(s, title, { x, y: 1.15, w: 4.5, h: 0.28, size: 11, bold: true, color: NAVY });
      table(s, [["質問", "答え"], ...rows], { x, y: 1.47, w: 4.5, colW: [1.75, 2.75], rowH: [0.3, 0.62, 0.62, 0.62, 0.62], size: 9, leftCols: [1], name: dark });
    };
    col(X0, "現場の方から", field, "qa-field");
    col(X0 + 4.7, "経営の方から", mgmt, "qa-mgmt");
    note(s, "値段は見積、削減時間は仮説。端末の条件（保護めがね・技適・国内販売）は試験導入の前に確認する");
  }

  // ---------- 12 まずは1拠点から ----------
  {
    const s = add(12);
    heading(s, "まずは1拠点から：3か月で、効果を数字で確かめる", "合格ラインを先に決めて、届かなければ本契約しない");
    table(s, [["試験導入の条件", ""], ["対象", "1拠点・2〜3班（職長がグラスを掛ける）"], ["期間", "3か月（準備1か月・測定2週間・使用4週間）"], ["費用", "導入支援 200万円（着手金50%）"], ["合格ライン", `1台1日${be}分以上の削減`], ["本契約", `1拠点10台で年300万円＋AIグラスのリース約${man(V.leaseSite())}万円`]],
      { x: X0, y: 1.15, w: 5.4, colW: [1.3, 4.1], rowH: 0.42, size: 10, leftCols: [1], name: "trial" });
    box(s, { x: 6.1, y: 1.15, w: 3.5, h: 2.52, fill: K.ink });
    txt(s, [run("現場の勘を", { bold: true, fontSize: 20, breakLine: true }), run("次の担い手へ", { bold: true, fontSize: 20, breakLine: true }), run("", { breakLine: true }),
      run("株式会社カンツギ（設立準備中）", { fontSize: 10, color: YELLOW, bold: true })], { x: 6.25, y: 1.15, w: 3.2, h: 2.52, color: K.paper, valign: "middle", psa: 2 });
    box(s, { x: X0, y: 3.85, w: W, h: 1.0, fill: PALE });
    txt(s, [lab("この資料について：", false), run("数字は事業計画の試算（値段は見積、削減時間は仮説、端末とリースは市販品の公表値を当てはめた仮定）。画面と写真台帳は架空の工事のデモ。計算の過程は「計算詳細版」と裏取りのExcelにまとめています", {})],
      { x: X0 + 0.14, y: 3.85, w: W - 0.28, h: 1.0, size: 9.5, valign: "middle" });
  }

  fs.mkdirSync(OUT_DIR, { recursive: true });
  await pres.writeFile({ fileName: OUT });
  console.log("wrote", OUT);
}

main().catch((e) => { console.error(e); process.exit(1); });
