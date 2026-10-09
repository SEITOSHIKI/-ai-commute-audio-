// Build the 令和の虎 pitch deck for GEN³ Works (三現ワークス): 15 slides following 谷本吉紹氏の事業計画書チェックポイント ①〜⑦.
// Usage: node build_deck.js [outDir]   (needs pptxgenjs and sharp)
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const sharp = require("sharp");
const T = require("../design-system/georec/pptx-theme.js");

const { color: K, size: S } = T;
const OUT_DIR = process.argv[2] || __dirname;
const OUT = path.join(OUT_DIR, "令和の虎_事業計画書_三現ワークス.pptx");
const LOGOS = path.join(__dirname, "../design-system/georec/logos");
const F = T.font.sans;
const X0 = 0.4, W = 9.2;

async function logo(name) {
  const buf = await sharp(path.join(LOGOS, name + ".svg"), { density: 600 }).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

// ---------- primitives (fresh option objects every call) ----------
function txt(slide, text, o) {
  slide.addText(text, {
    fontFace: F, fontSize: o.size || S.body, color: o.color || K.ink, bold: !!o.bold,
    x: o.x, y: o.y, w: o.w, h: o.h, margin: o.margin ?? 0, valign: o.valign || "top",
    align: o.align || "left", lineSpacingMultiple: o.lsm || 1.15, paraSpaceAfter: o.psa || 0,
    fill: o.fill ? { color: o.fill } : undefined, isTextBox: true, objectName: o.name,
  });
}
function box(slide, o) {
  slide.addShape("rect", {
    x: o.x, y: o.y, w: o.w, h: o.h,
    fill: { color: o.fill || K.paper },
    line: o.line ? { color: o.line, width: o.lw || T.stroke.rule } : { type: "none" },
    objectName: o.name,
  });
}
function hline(slide, x, y, w, c, lw) {
  slide.addShape("line", { x, y, w, h: 0, line: { color: c, width: lw } });
}
const run = (text, o = {}) => ({ text, options: { fontFace: F, ...o } });

const ACC = "B8692E"; // copper: the conductor of a cable; the one accent of the pitch deck
function heading(slide, title, lead) {
  const m = /^([①②③④⑤⑥⑦])\s*(.*)$/.exec(title);
  if (m) title = [run(m[1] + " ", { color: ACC }), run(m[2])];
  txt(slide, title, { x: X0, y: 0.28, w: 8.7, h: 0.42, size: S.slideTitle, bold: true, valign: "middle", name: "title" });
  if (lead) txt(slide, lead, { x: X0, y: 0.72, w: 8.7, h: 0.28, size: S.lead, color: K.gray700, valign: "middle", name: "lead" });
}

// Simple ruled table: header on fill with strong ink rule, body rows with light rules.
function table(slide, rows, o) {
  const fs = o.size || S.table;
  const body = rows.map((r, i) =>
    r.map((c, j) => {
      const cell = typeof c === "object" ? c : { text: String(c) };
      const head = i === 0;
      const strong = o.strongRows && o.strongRows.includes(i);
      const em = o.emCol !== undefined && j === o.emCol;
      return {
        text: cell.text,
        options: {
          fontFace: F, fontSize: fs, color: (o.mutedCol === j && !head) ? K.gray700 : K.ink,
          bold: head || strong || !!cell.bold,
          align: j === 0 || (o.leftCols || []).includes(j) ? "left" : (o.align || "right"),
          valign: "middle",
          fill: { color: head || em ? K.fill : K.paper },
          margin: [0.02, 0.06, 0.02, 0.06],
          border: [
            { type: strong ? "solid" : "none", color: K.ink, pt: T.stroke.strong },
            { type: "none" },
            head ? { type: "solid", color: K.ink, pt: T.stroke.strong } : { type: "solid", color: K.rule, pt: T.stroke.rule },
            { type: "none" },
          ],
        },
      };
    })
  );
  slide.addTable(body, { x: o.x, y: o.y, w: o.w, colW: o.colW, rowH: o.rowH, objectName: o.name });
}


const TAGLINE = "現場の勘を、次の担い手へ。";

// ---------- notes: time slot, the slide's one message, then the script ----------
const NOTE_PARTS = {
  1: ["0:00 - 0:25", "人が減っても現場が迷わず回る仕組みをつくる。希望金額は500万円",
    "株式会社三現ワークスです。三現とは、現場・現物・現実のこと。熟練者の勘が退職とともに消えていく現場に、スマートグラスで記録して次の担い手とAIに渡す仕組みをつくります。希望金額は500万円です。"],
  2: ["0:25 - 1:00", "スマートグラスで現場の作業を記録し、報告と教育を自動にするサービス",
    "事業の概要です。何のビジネスか。作業者がスマートグラスを着けるだけで作業を記録し、報告書づくりと若手の教育を自動にするサービスです。既存のサービスとの違いは、記録を貯めるだけでなく、現物に重ねて渡し、教材にまですること。誰の悩みを解くか。人手不足の現場で、報告と育成に追われる管理者とDX担当者です。収益は、1拠点あたり年300万円の利用料と、初年度の導入支援です。"],
  3: ["1:00 - 1:45", "報告と育成に追われ、熟練者が辞めると現場が止まる",
    "対象のお客様は三人です。一人目は、現場で孤軍奮闘するDX推進担当者。ツールを入れても現場に定着せず、成果を説明できない。二人目は、人手不足の製造現場で働く作業者。日報や写真整理に毎月何時間も取られ、聞ける熟練者もいない。三人目は管理者や施工管理者。立会に行けず、後継者もいない。国の調査でも、技能継承の主流は、ベテランに居続けてもらうことで54.8%。デジタル化で継承が進んだ企業は8.7%しかありません。"],
  4: ["1:45 - 2:30", "記録は自動、渡す先は三つ。データが貯まるほど真似されにくくなる",
    "解決策です。作業者の一人称の動画と会話を撮り続け、AIが整理し、若手には現物に重ねて、管理者には報告書として、次の担い手には教材として渡します。真似されにくい理由は三つ。一つ目、図面に現れない現場処置の記録は、使うほど貯まり、後から来た会社には移せません。二つ目、映像のどこが重要かを見分ける意味づけは、現場経験者にしかできません。三つ目、電線工場の工程ごとの手順と報告書の型を、業界で最初に揃えます。"],
  5: ["2:30 - 3:15", "デモ：図面との差異を見つけ、報告書が自動でできる",
    "実際の画面です。ケーブル接続の現場で、若手が処理した半導電層が図面より8ミリ短いことを検出しました。遠隔の熟練者が、テープの位置から測り直すよう指示します。その判断も記録されます。管理者の画面では、手順ごとに写真が自動で割り当てられ、報告書ができていきます。作業者が報告書に使った時間はゼロです。"],
  6: ["3:15 - 3:50", "「貯める」会社は現れた。当社は「渡す」で分かれる",
    "競合です。記録を貯める会社は、東大発のAirion、キャディのCADDiなど、すでにあります。正直に言えば、貯めるところは同じ設計です。分かれるのは渡し方と対象です。連続工程の現場で、現物に重ねて渡し、教材にまでする会社はまだありません。最大の脅威はCADDiですが、図面に現れない現場の処置は、当社にしか貯まりません。"],
  7: ["3:50 - 4:30", "狙う市場は約100億円。周辺市場は年7〜9%で伸びている",
    "市場規模です。土台となる国内のスマートファクトリー市場は、2025年の42億ドルから年9%で伸びる見込みです。建設テック市場も、建築分野だけで2030年度に約3,040億円、年7.4%の成長です。当社が狙うのは、電線・ケーブル工場の353事業所と、中堅以上の電気工事会社約3,000社。年300万円をかけると約100億円の市場です。5年目に85拠点、シェア2.5%を取りに行きます。"],
  8: ["4:30 - 5:15", "1拠点の生涯粗利は約1,865万円、獲得コストは約64万円",
    "お金の流れです。お客様は1拠点あたり年300万円、初年度だけ導入支援200万円を払います。お客様は約2.1か月で元が取れます。1拠点が生む生涯の粗利、LTVは、年255万円を保守的に7年分と、導入支援の粗利を足して約1,865万円。一方、1拠点を獲得する費用、CACは、5年目の営業人件費と広告費を新規拠点数で割って約64万円です。獲得費用は3か月で回収できます。"],
  9: ["5:15 - 5:55", "売上は拠点数×月25万円で積み上がる。5年目は月2,400万円",
    "売上と費用です。売上は、拠点数かける月25万円の利用料が積み上がる形です。5年目の月次売上は約2,400万円で、内訳は利用料が約1,790万円、導入支援が約550万円です。費用の中心は人件費で、1人あたり年800万円。所在地は川崎市を考えています。京浜の工場地帯と羽田・新幹線に近く、関東の電線工場と東京の電気工事会社の両方に出やすい場所です。家賃は1年目15坪で月20万円、5年目40坪で月50万円、広告と展示会は1年目200万円から5年目500万円を見込んでいます。"],
  10: ["5:55 - 6:25", "4年目に損益分岐点を超え、5年目の安全余裕率は32%",
    "損益分岐点です。販管費を粗利率で割った損益分岐点売上は、4年目で1億5,100万円。売上1億9,000万円がこれを初めて上回ります。5年目は分岐点1億9,600万円に対して売上2億9,000万円で、安全余裕率は32%です。"],
  11: ["6:25 - 7:00", "黒字倒産しない。最低残高は固定費3か月分",
    "資金計画です。必要資金は1.5億円。今回の500万円で準備期を走り、実証の結果をもって日本政策金融公庫の創業融資1,000万円、CVCとVCから1億1,500万円を調達します。現金残高は最も薄い3年目でも6,000万円で、固定費3か月分を下回りません。利用料は年額前払い、導入支援は着手金をいただき、黒字倒産を防ぎます。"],
  12: ["7:00 - 7:40", "チームは現場と技術の両方を揃える。顧問とパートナーで弱みを補う",
    "チームです。私は電線メーカーで地中送電線の施工管理を5年やり、独学でAIの仕組みを作ってきました。毎朝ニュースを音声で配信する仕組みも、個人で作って動かしています。ただ、私一人では足りません。技術の責任者、生産技術の経験者、現場出身のカスタマーサクセスを揃え、電線業界の顧問、スマートグラスのメーカー、大学と組みます。"],
  13: ["7:40 - 8:00", "導入期・成長期を越え、成熟期には上場を目指す",
    "成長の道筋です。導入期は記録と報告の自動化で顧客をつかみ、成長期は図面との照合と品質管理で黒字化し、成熟期はフィジカルAIと海外展開で上場を目指します。"],
  14: ["8:00 - 8:30", "500万円で、6か月以内に最初の1拠点を動かす",
    "500万円の使い道です。試作品の開発に250万円、端末に80万円、1拠点での実証に100万円、設立と契約に40万円、予備30万円。私の給料はゼロです。3か月で試作品、6か月で実証の結果を出します。条件は10%でのご提案です。"],
  15: ["8:30 - 9:00", "最大のリスクは自分。だから役割と判断基準を先に決めておく",
    "最後にリスクです。最大のリスクは、私自身が律速になること。だから役割を絞り、判断の基準を先に決めて、私がいなくても回る会社にします。日本の品質を支えてきた現場の勘を、人に依存しない形で残したい。その最初の一歩に、500万円をお願いします。"],
};
const NOTES = Object.fromEntries(Object.entries(NOTE_PARTS).map(([n, [t, msg, body]]) => [n, `【${t}】\n要約：${msg}\n\n${body}`]));
const ASSETS = path.join(__dirname, "assets");
const chap = (n, t) => `${n} ${t}`;


async function main() {
  const pres = new pptxgen();
  pres.layout = T.slide.layout;
  pres.title = "株式会社三現ワークス 事業計画書";
  pres.theme = { headFontFace: F, bodyFontFace: F };

  const markBlack = await logo("gen3-mark-black");
  const markWhite = await logo("gen3-mark-white");
  const wordWhite = await logo("gen3-logotype-reverse");
  const logoColor = await logo("gen3-logotype-color");

  pres.defineSlideMaster({
    title: "GR_CONTENT",
    background: { color: K.paper },
    objects: [
      { image: { x: 8.15, y: 5.2, w: 1.45, h: 1.45 * 120 / 560, data: logoColor } },
      { text: { text: TAGLINE, options: { x: 0.8, y: 5.29, w: 3.5, h: 0.22, fontFace: F, fontSize: S.footer, color: K.gray500, align: "left", margin: 0, valign: "middle" } } },
    ],
    slideNumber: { x: 0.4, y: 5.29, w: 0.35, h: 0.22, fontFace: F, fontSize: S.footer, color: K.gray500, align: "left", margin: 0 },
  });
  pres.defineSlideMaster({ title: "GR_COVER", background: { color: K.paper }, objects: [{ image: { x: 8.15, y: 5.2, w: 1.45, h: 1.45 * 120 / 560, data: logoColor } }] });
  const add = (n) => { const s = pres.addSlide({ masterName: n === 1 ? "GR_COVER" : "GR_CONTENT" }); s.addNotes(NOTES[n]); return s; };

  // ---------- Slide 1 表紙・希望金額 ----------
  {
    const s = add(1);
    box(s, { x: 0, y: 0, w: 3.6, h: 5.625, fill: K.ink, name: "cover-panel" });
    s.addImage({ data: wordWhite, x: 0.35, y: 1.05, w: 3.0, h: 3.0 * 120 / 560 });
    txt(s, "現場の勘を、\n次の担い手へ。", { x: 0.4, y: 2.15, w: 3.0, h: 0.85, size: 20, bold: true, color: K.paper, lsm: 1.2 });
    txt(s, "株式会社三現ワークス（事業計画）", { x: 0.4, y: 3.2, w: 3.0, h: 0.3, size: 11, color: K.paper });
    txt(s, "三現＝現場・現物・現実", { x: 0.4, y: 3.52, w: 3.0, h: 0.28, size: 9.5, color: K.paper });
    const rx = 4.0, rw = 5.3;
    txt(s, "事業計画書", { x: rx, y: 0.55, w: rw, h: 0.28, size: S.lead, color: K.gray700 });
    txt(s, "人が減っても、\n現場が迷わず回る仕組みをつくる", { x: rx, y: 0.9, w: rw, h: 0.85, size: 22, bold: true, lsm: 1.2 });
    txt(s, "スマートグラスで熟練者の作業を記録し、報告を自動にし、次の担い手とAIの教材にする", { x: rx, y: 1.85, w: rw, h: 0.5, size: 11.5 });
    box(s, { x: rx, y: 2.6, w: rw, h: 1.3, line: ACC, lw: 2, name: "ask" });
    txt(s, "希望金額", { x: rx + 0.2, y: 2.68, w: 2, h: 0.3, size: 12, bold: true, color: K.gray700 });
    txt(s, "500万円", { x: rx + 0.2, y: 2.95, w: 3.2, h: 0.8, size: 44, bold: true, color: ACC, valign: "middle" });
    txt(s, "準備期6か月の資金\n（出資）", { x: rx + 3.45, y: 3.05, w: 1.75, h: 0.6, size: 10, color: K.gray700, valign: "middle" });
    txt(s, "志願者：＿＿＿＿＿＿＿＿＿＿", { x: rx, y: 4.25, w: rw, h: 0.3, size: 10.5, color: K.gray700 });
  }

  // ---------- Slide 2 ①事業概要 ----------
  {
    const s = add(2);
    heading(s, chap("①", "事業概要 ─ 現場の作業を記録し、報告と教育を自動に"), "スマートグラスで撮るだけ。報告書・図面との照合・教材までを一続きで届ける");
    const q = [
      ["何のビジネス？", "作業者がスマートグラスやヘルメットカメラを着けるだけで、一人称の動画と会話を記録。AIが報告書を作り、図面と照合し、教材にするサービス"],
      ["既存の市場との違い", "記録・手順表示・教育のツールは別々にある。当社は「記録→照合→報告→教材」を一続きで提供し、現場経験者が中身を監修する"],
      ["誰のどんな悩みを解決する？", "人手不足の製造・電気工事の現場で、報告と育成に追われる管理者・DX担当者。熟練者が辞めると判断が消え、手戻りと不良が増える"],
      ["収益構造", "1拠点あたり年300万円の利用料（基盤＋ライン数＋追加機能）が毎年積み上がる。初年度だけ導入支援200万円"],
    ];
    const cw = (W - 0.15) / 2, ch = 1.62, top = 1.15;
    q.forEach(([h, v], i) => {
      const x = X0 + (i % 2) * (cw + 0.15), y = top + Math.floor(i / 2) * (ch + 0.15);
      box(s, { x, y, w: cw, h: ch, fill: i === 3 ? K.ink : K.fill, name: "summary-" + i });
      const c = i === 3 ? K.paper : K.ink;
      txt(s, h, { x: x + 0.18, y: y + 0.12, w: cw - 0.36, h: 0.34, size: 13, bold: true, color: c });
      txt(s, v, { x: x + 0.18, y: y + 0.52, w: cw - 0.36, h: ch - 0.62, size: 11, color: c });
    });
    txt(s, "対象：電線・ケーブル工場（連続工程）から始め、製品の出口である電気工事会社へ広げる", { x: X0, y: 4.75, w: W, h: 0.3, size: 10, color: K.gray700 });
  }

  // ---------- Slide 3 ②課題（ペイン） ----------
  {
    const s = add(3);
    heading(s, chap("②", "課題 ─ 報告と育成に追われ、熟練者が辞めると止まる"), "ターゲットは三者。それぞれに不満・不便・損失がある");
    const ps = [
      ["DX推進担当者", "現場で孤軍奮闘", ["ツールを入れても現場に定着しない", "成果を数字で説明できず、予算が続かない", "現場とシステムの言葉をつなぐ人がいない"]],
      ["現場の作業者", "人手不足の製造業のブルーカラー", ["日報・写真整理・報告書に毎月何時間も取られる", "聞ける熟練者がいない。手順書に勘どころが書いていない", "ミスに気づくのが後になり、手戻りが出る"]],
      ["管理者・施工管理者", "引退と後継者不足", ["立会に行けず、現場の状況が見えない", "熟練者の判断が退職とともに消える", "育成する時間がなく、品質が人に依存する"]],
    ];
    const cw = 2.97, gap = 0.145, top = 1.12, ch = 2.35;
    ps.forEach(([who, sub, items], i) => {
      const x = X0 + i * (cw + gap);
      box(s, { x, y: top, w: cw, h: ch, line: K.rule, name: "persona-" + i });
      box(s, { x, y: top, w: cw, h: 0.55, fill: K.ink });
      txt(s, [run(who, { bold: true, fontSize: 12.5, breakLine: true }), run(sub, { fontSize: 9 })], { x: x + 0.14, y: top + 0.02, w: cw - 0.28, h: 0.52, color: K.paper, valign: "middle" });
      txt(s, items.map((v, j) => run("・" + v, { breakLine: j < items.length - 1 })), { x: x + 0.14, y: top + 0.65, w: cw - 0.28, h: ch - 0.75, size: 10, psa: 5 });
    });
    const sy = 3.62;
    [["54.8%", "技能継承の取組で最多は「ベテランに居続けてもらう」(1)"], ["8.7%", "デジタル活用の効果として技能継承が円滑になった企業 (2)"], ["62.8%", "製造業で「指導する人材が不足」(1)"], ["月6時間", "1人あたりの書類作業の削減余地（建設業の電子化実績を準用）"]].forEach(([f, l], i) => {
      const cw2 = (W - 0.3) / 4, x = X0 + i * (cw2 + 0.1);
      box(s, { x, y: sy, w: cw2, h: 1.2, fill: K.fill, name: "pain-stat-" + i });
      txt(s, f, { x: x + 0.1, y: sy + 0.08, w: cw2 - 0.2, h: 0.45, size: 20, bold: true, valign: "middle" });
      txt(s, l, { x: x + 0.1, y: sy + 0.55, w: cw2 - 0.2, h: 0.6, size: 8.5 });
    });
    txt(s, "(1) 経済産業省・厚生労働省・文部科学省「2026年版 ものづくり白書」（令和8年5月）　(2) 労働政策研究・研修機構 調査シリーズNo.265（2026年3月）", { x: X0, y: 4.98, w: W, h: 0.26, size: S.note, color: K.gray500 });
  }

  // ---------- Slide 4 ③解決策・真似されにくい仕組み ----------
  {
    const s = add(4);
    heading(s, chap("③", "解決策 ─ 記録は自動。貯まるほど真似されにくい"), "一人称の動画と会話をAIが整理し、若手・管理者・次の担い手に渡す。その先にフィジカルAI");
    s.addImage({ path: path.join(ASSETS, "shot-overview.png"), x: X0, y: 1.15, w: 5.6, h: 5.6 / 2.375 });
    txt(s, "サービス内容", { x: X0, y: 3.62, w: 5.6, h: 0.26, size: 10.5, bold: true });
    txt(s, [
      run("若手へ：", { bold: true }), run("図面・作業手順を現物に重ね、差異をその場で表示", { breakLine: true }),
      run("管理者へ：", { bold: true }), run("報告書の手順に沿って写真を抽出し、当日の報告書を自動作成", { breakLine: true }),
      run("次の担い手へ：", { bold: true }), run("勘どころを解説動画・音声番組・研修資料にして当社が届ける"),
    ], { x: X0, y: 3.9, w: 5.6, h: 0.95, size: 9.5, psa: 3 });
    const rx = 6.2, rw = 3.4;
    box(s, { x: rx, y: 1.15, w: rw, h: 3.7, fill: K.ink, name: "moat" });
    txt(s, "真似されにくい3つの仕組み", { x: rx + 0.15, y: 1.22, w: rw - 0.3, h: 0.3, size: 11.5, bold: true, color: K.paper });
    [
      ["記録が貯まるほど強くなる", "図面に現れない現場処置の記録は使うほど増え、後発には移せない"],
      ["意味づけは現場経験者にしかできない", "映像のどこが重要かを見分けて意味を付けるのは、施工・製造の経験者"],
      ["業界の型を先に揃える", "電線工場の工程ごとの手順と報告書の型を、業界で最初に揃える"],
    ].forEach(([h, d], i) => {
      const y = 1.62 + i * 1.05;
      txt(s, [run((i + 1) + ". " + h, { bold: true, breakLine: true, fontSize: 10.5 }), run(d, { fontSize: 9 })], { x: rx + 0.15, y, w: rw - 0.3, h: 0.98, color: K.paper, psa: 2 });
    });
    txt(s, "ノウハウの核：判断ラベル付きの現場データ（どの手順の、どの場面が、なぜ重要か）", { x: X0, y: 4.98, w: W, h: 0.26, size: S.note, color: K.gray500 });
  }

  // ---------- Slide 5 デモ ----------
  {
    const s = add(5);
    heading(s, "③ デモ ─ 図面との差異を見つけ、報告書が自動でできる", "架空のケーブル接続工事。若手の作業を記録し、8mmの差異を検出、報告書と教材までつながる");
    txt(s, "① 作業者のスマートグラス", { x: X0, y: 1.1, w: 5.4, h: 0.24, size: 10, bold: true });
    s.addImage({ path: path.join(ASSETS, "shot-glass.png"), x: X0, y: 1.36, w: 5.4, h: 5.4 / 2.382 });
    txt(s, "② 管理者への自動報告", { x: 6.0, y: 1.1, w: 3.6, h: 0.24, size: 10, bold: true });
    s.addImage({ path: path.join(ASSETS, "shot-manager.png"), x: 6.0, y: 1.36, w: 3.6, h: 3.6 / 1.647 });
    const by = 1.36 + 5.4 / 2.382 + 0.15;
    [
      ["検出", "半導電層の端部 実測112mm／図面120mm（8mm不足）"],
      ["是正", "遠隔の熟練者が「テープ基準で測り直す」と指示。判断として記録"],
      ["報告", "手順ごとに写真を自動割当。作業者の報告書作成時間は0分"],
    ].forEach(([k, v], i) => {
      const y = by + i * 0.3;
      box(s, { x: X0, y: y + 0.03, w: 0.5, h: 0.24, fill: K.ink });
      txt(s, k, { x: X0, y: y + 0.03, w: 0.5, h: 0.24, size: 9, bold: true, color: K.paper, align: "center", valign: "middle" });
      txt(s, v, { x: X0 + 0.6, y, w: 4.8, h: 0.3, size: 9.5, valign: "middle" });
    });
    txt(s, "デモアプリ：claude.ai 上で再生可能（作業の再生・1件ずつ送り・全体像の表示）。寸法・人物はすべて架空", { x: X0, y: 4.98, w: W, h: 0.26, size: S.note, color: K.gray500 });
  }

  // ---------- Slide 5 競合（貯める／渡す） ----------
  {
    const s = add(6);
    heading(s, "③ 競合 ─ 「貯める」会社は現れた。当社は「渡す」で分かれる", "判断の理由を扱う会社は現れた。連続工程の現場で、現物に重ねて渡し、教材にまでする会社はまだない");
    // positioning map
    const mx = X0, my = 1.15, mw = 4.35, mh = 3.6;
    txt(s, "ポジショニングマップ", { x: mx, y: my, w: mw, h: 0.26, size: 10.5, bold: true });
    const px = mx + 0.3, py = my + 0.42, pw = mw - 0.4, ph = mh - 0.9;
    box(s, { x: px + pw / 2, y: py, w: pw / 2, h: ph / 2, fill: K.fill, name: "target-quadrant" });
    hline(s, px, py + ph / 2, pw, K.gray500, T.stroke.rule);
    s.addShape("line", { x: px + pw / 2, y: py, w: 0, h: ph, line: { color: K.gray500, width: T.stroke.rule } });
    txt(s, "仕組み（プロダクト）", { x: px + pw / 2 - 1.0, y: py - 0.24, w: 2.0, h: 0.2, size: 8.5, color: K.gray700, align: "center" });
    txt(s, "人の時間（労働集約）", { x: px + pw / 2 - 1.0, y: py + ph + 0.04, w: 2.0, h: 0.2, size: 8.5, color: K.gray700, align: "center" });
    txt(s, "← 実績・手順", { x: px, y: py + ph / 2 + 0.03, w: 1.4, h: 0.2, size: 8, color: K.gray700 });
    txt(s, "判断の理由 →", { x: px + pw - 1.4, y: py + ph / 2 + 0.03, w: 1.4, h: 0.2, size: 8, color: K.gray700, align: "right" });
    const dots = [
      ["MES・ERP", 0.1, 0.12, false, "r"],
      ["tebiki・Teachme Biz（器）", 0.08, 0.32, false, "r"],
      ["AR作業支援", 0.34, 0.43, false, "l"],
      ["CADDi（組立・図面）", 0.58, 0.1, false, "r"],
      ["Airion（貯めるまで）", 0.7, 0.36, false, "r"],
      ["三現ワークス", 0.9, 0.24, true, "l"],
      ["三菱総研 匠AI", 0.62, 0.7, false, "r"],
      ["社内の熟練者（属人）", 0.86, 0.88, false, "l"],
    ];
    dots.forEach(([label, fx, fy, ours, side]) => {
      const cx = px + fx * pw, cy = py + fy * ph, r = ours ? 0.11 : 0.075;
      s.addShape("ellipse", { x: cx - r, y: cy - r, w: 2 * r, h: 2 * r, fill: { color: ours ? K.ink : K.paper }, line: { color: K.ink, width: 1.25 }, objectName: "pos-" + label });
      const o = { size: ours ? 10 : 8, bold: ours, valign: "middle", h: 0.22, w: 1.7 };
      if (side === "r") txt(s, label, { ...o, x: cx + r + 0.04, y: cy - 0.11, align: "left" });
      else txt(s, label, { ...o, x: cx - r - 1.74, y: cy - 0.11, align: "right" });
    });

    // store / hand-off table
    const sx = 4.95, sw = 4.65;
    txt(s, "貯める（記録・構造化）と、渡す（教材にして届ける）", { x: sx, y: my, w: sw, h: 0.26, size: 10.5, bold: true });
    const cols = [1.5, 1.1, 1.1, 0.95];
    const head = ["会社・サービス", "貯める", "渡す", "対象"];
    const rows = [
      ["Airion 技能継承くん", "○ AI対話", "× 顧客任せ", "業種横断"],
      ["キャディ CADDi", "○ 判断履歴", "×", "組立・図面"],
      ["三菱総研 匠AI", "○ コンサル型", "×", "個別案件"],
      ["FRONTEO KIBIT", "△ 既存文書", "×", "業種横断"],
      ["tebiki／Teachme Biz", "×", "△ 器のみ", "業種横断"],
      ["AR作業支援（各社）", "×", "△ 手順表示", "業種横断"],
      [{ text: "三現ワークス", bold: true }, { text: "○ 常時収録", bold: true }, { text: "○ 現物照合＋教材", bold: true }, { text: "連続工程", bold: true }],
    ];
    table(s, [head, ...rows], { x: sx, y: my + 0.3, w: sw, colW: cols, rowH: 0.27, size: 8, align: "left", strongRows: [7], name: "store-handoff" });
    const ty = my + 0.3 + 8 * 0.27 + 0.08;
    box(s, { x: sx, y: ty, w: sw, h: 0.66, fill: K.ink, name: "threat" });
    txt(s, [
      run("最大の脅威：", { bold: true }), run("CADDiが連続工程に降りてくること。課題認識は同じ（「製造業の叡智の8割以上は暗黙知」）", { breakLine: true }),
      run("防御線：", { bold: true }), run("図面に現れない現場処置の蓄積と、現場で監修できる人"),
    ], { x: sx + 0.12, y: ty, w: sw - 0.24, h: 0.66, size: 8.5, color: K.paper, valign: "middle", psa: 2 });
    txt(s, [
      run("STP：", { bold: true }), run("工程の形（連続／組立／施工）×規模で区分し、中堅の電線工場を狙う", { breakLine: true }),
      run("競争戦略：", { bold: true }), run("ニッチャー（コトラー）×差別化集中（ポーター）。大手の内製は個別開発で中堅に届かない"),
    ], { x: sx, y: ty + 0.7, w: sw, h: 0.46, size: 8, psa: 1 });
    txt(s, "出典：各社公表資料（Airion 2025年7月、キャディ 2026年8月6日、三菱総合研究所、FRONTEO、tebiki、スタディスト）。詳細と大手の内製事例は補足資料", { x: X0, y: 5.04, w: W, h: 0.24, size: S.note, color: K.gray500 });
  }
  // ---------- Slide 7 ④市場規模・成長性 ----------
  {
    const s = add(7);
    heading(s, chap("④", "市場規模 ─ 狙う市場は約100億円、周辺は年7〜9%成長"), "TAM（周辺の大きな市場）→ SAM（当社が届く市場）→ SOM（5年目に取る市場）");
    const cx = 2.55, cy = 3.05;
    s.addShape("ellipse", { x: cx - 2.1, y: cy - 1.85, w: 4.2, h: 3.7, fill: { color: K.fill }, line: { color: K.gray500, width: 1 }, objectName: "tam" });
    s.addShape("ellipse", { x: cx - 1.3, y: cy - 0.75, w: 2.6, h: 2.5, fill: { color: K.gray700 }, line: { type: "none" }, objectName: "sam" });
    s.addShape("ellipse", { x: cx - 0.6, y: cy + 0.35, w: 1.2, h: 1.2, fill: { color: K.ink }, line: { type: "none" }, objectName: "som" });
    txt(s, [run("TAM", { bold: true, breakLine: true }), run("スマートファクトリー＋建設テック", { fontSize: 8.5 })], { x: cx - 1.6, y: cy - 1.7, w: 3.2, h: 0.6, size: 11, align: "center" });
    txt(s, [run("SAM 約100億円", { bold: true, breakLine: true }), run("約3,350拠点×年300万円", { fontSize: 8.5 })], { x: cx - 1.2, y: cy - 0.6, w: 2.4, h: 0.6, size: 11, color: K.paper, align: "center" });
    txt(s, [run("SOM", { bold: true, breakLine: true }), run("2.55億円", { fontSize: 9 })], { x: cx - 0.6, y: cy + 0.55, w: 1.2, h: 0.8, size: 10.5, color: K.paper, align: "center", valign: "middle" });
    const rx = 5.15, rw = 4.45;
    table(s, [
      ["区分", "規模", "根拠"],
      ["TAM 製造", "42億→92億米ドル", "国内スマートファクトリー市場 2025→2034年 (1)"],
      ["TAM 建設", "1,845億円→3,043億円", "国内建設テック市場（建築分野）2023→2030年度 (2)"],
      ["SAM", "約100億円", "電線・ケーブル製造353事業所 (3)＋中堅以上の電気工事会社 約3,000社 (4)、×年300万円"],
      ["SOM", "2.55億円（シェア2.5%）", "5年目85拠点×年300万円"],
    ], { x: rx, y: 1.15, w: rw, colW: [0.8, 1.45, 2.2], rowH: 0.42, size: 8, leftCols: [1, 2], name: "market" });
    box(s, { x: rx, y: 3.55, w: rw, h: 1.1, fill: K.ink, name: "growth" });
    txt(s, "市場の伸び率", { x: rx + 0.15, y: 3.6, w: rw - 0.3, h: 0.28, size: 10.5, bold: true, color: K.paper });
    txt(s, [
      run("国内スマートファクトリー：年9.03%成長（2026〜2034年予測）(1)", { breakLine: true }),
      run("国内建設テック（建築分野）：年7.4%成長（2023→2030年度予測）(2)", { breakLine: true }),
      run("追い風：内閣府「人工知能基本計画」が暗黙知の多い現場へのAI実装を推進"),
    ], { x: rx + 0.15, y: 3.9, w: rw - 0.3, h: 0.72, size: 8.5, color: K.paper, psa: 2 });
    txt(s, "(1) IMARC Group「Japan Smart Factory Market」　(2) 矢野経済研究所 プレスリリースNo.3789（2025年4月）建築分野ソフトウェア　(3) 経済産業省「経済構造実態調査」2024年　(4) 電気工事業の許可業者65,497社（令和7年3月末、国土交通省）から推計", { x: X0, y: 4.9, w: W, h: 0.3, size: S.note, color: K.gray500 });
  }

  // ---------- Slide 4 収益の仕組み ----------
  {
    const s = add(8);
    heading(s, "⑤ ビジネスモデル ─ 年300万円、約2か月で元が取れる", "当社の収入は、毎年積み上がる利用料と、初年度だけの導入支援の二本");
    const lx = X0, lw = 4.45, top = 1.15;
    txt(s, "1拠点が払う金額（万円）", { x: lx, y: top, w: lw, h: 0.26, size: 10.5, bold: true });
    const k = (lw - 1.05) / 500;
    const segs = [
      ["基盤", 100, K.ink, K.paper],
      ["ライン登録", 100, K.gray700, K.paper],
      ["追加機能", 100, K.gray500, K.ink],
      ["導入支援", 200, K.paper, K.ink],
    ];
    [["初年度", 500, segs], ["2年目以降", 300, segs.slice(0, 3)]].forEach(([lab, total, ss], r) => {
      const y = top + 0.36 + r * 0.56;
      txt(s, lab, { x: lx, y, w: 0.85, h: 0.42, size: 9.5, bold: true, valign: "middle" });
      let x = lx + 0.85;
      ss.forEach(([name, v, fill, ink]) => {
        box(s, { x, y, w: v * k, h: 0.42, fill, line: K.ink, lw: 0.75, name: "seg-" + name + r });
        txt(s, name + " " + v, { x, y, w: v * k, h: 0.42, size: 8.5, bold: true, color: ink, align: "center", valign: "middle" });
        x += v * k;
      });
      txt(s, String(total), { x: x + 0.04, y, w: 0.4, h: 0.42, size: 11, bold: true, valign: "middle" });
    });
    const dy = top + 1.55;
    [
      ["基盤", "100万円／年", "アプリと現場データ基盤の利用料（1拠点）"],
      ["ライン登録", "20万円×5ライン", "記録の対象にする生産ライン・工区の数に比例"],
      ["追加機能", "50万円×2本", "教材制作（KNACK）・危険表示（SAFE）など、選んだ数に比例"],
      ["導入支援", "200万円（初年度のみ）", "当社の担当者が約2か月、現場に入る作業"],
    ].forEach(([n, p, d], i) => {
      const y = dy + i * 0.36;
      hline(s, lx, y, lw, K.rule, T.stroke.rule);
      txt(s, [run(n, { bold: true, breakLine: true }), run(p, { fontSize: 8, color: K.gray700 })], { x: lx, y: y + 0.02, w: 1.45, h: 0.34, size: 9, valign: "middle" });
      txt(s, d, { x: lx + 1.5, y, w: lw - 1.5, h: 0.36, size: 9, valign: "middle" });
    });
    hline(s, lx, dy + 4 * 0.36, lw, K.rule, T.stroke.rule);
    box(s, { x: lx, y: dy + 1.52, w: lw, h: 0.72, fill: K.ink });
    txt(s, [run("LTV 約1,865万円", { bold: true }), run("（年粗利255万円×保守的に7年＋導入支援の粗利80万円）", { breakLine: true }), run("CAC 約64万円", { bold: true }), run("（5年目：営業人件費1,600万円＋広告500万円÷新規33拠点）", { breakLine: true }), run("LTV/CAC 約29倍・獲得費用は約3か月で回収", { bold: true })], { x: lx + 0.12, y: dy + 1.52, w: lw - 0.24, h: 0.72, size: 8, color: K.paper, valign: "middle", psa: 1 });

    // ROI line chart (customer view)
    const rx = 5.05, rw = 4.55;
    txt(s, "顧客から見た回収（対象者100名の拠点・万円）", { x: rx, y: top, w: rw, h: 0.26, size: 10.5, bold: true });
    const months = Array.from({ length: 13 }, (_, m) => String(m));
    const effect = Array.from({ length: 13 }, (_, m) => m * 240);
    const cost = Array.from({ length: 13 }, () => 500);
    const axis = { catAxisLabelColor: K.gray700, valAxisLabelColor: K.gray700, catAxisLabelFontSize: 8, valAxisLabelFontSize: 8,
      catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", valGridLine: { color: K.rule, size: 0.5 }, catGridLine: { style: "none" },
      valAxisMinVal: 0, valAxisMaxVal: 3000, valAxisMajorUnit: 500, catAxisLineShow: true, valAxisLineShow: false, showLegend: false };
    s.addChart([
      { type: pres.charts.LINE, data: [{ name: "浮いた工数の累計", labels: months, values: effect }], options: { chartColors: [K.ink], lineSize: 3, lineDataSymbol: "circle", lineDataSymbolSize: 5 } },
      { type: pres.charts.LINE, data: [{ name: "初年度費用", labels: months, values: cost }], options: { chartColors: [K.gray500], lineSize: 2, lineDash: "dash", lineDataSymbol: "none" } },
    ], { x: rx, y: top + 0.28, w: rw, h: 2.62, ...axis, objectName: "roi-chart" });
    txt(s, "← 約2.1か月で回収", { x: rx + 1.05, y: top + 2.3, w: 1.6, h: 0.22, size: 9.5, bold: true });
    txt(s, "初年度費用 500万円", { x: rx + 2.85, y: top + 2.3, w: 1.6, h: 0.22, size: 8.5, color: K.gray700 });
    txt(s, "浮いた工数の累計：12か月で2,880万円（5.8倍）", { x: rx + 0.65, y: top + 0.4, w: 3.6, h: 0.26, size: 10, bold: true });
    txt(s, "横軸は経過月数。時間単価4,000円（780万円÷1,950h）×月6h×100名＝月240万円。月6hは日報・進捗報告などの書類作業の削減分で、建設業の書類電子化実績（月6.5h）を準用した仮定", { x: rx, y: top + 2.94, w: rw, h: 0.5, size: S.note, color: K.gray700 });
  }

  // ---------- Slide 9 ⑥売上予測・費用 ----------
  {
    const s = add(9);
    heading(s, chap("⑥", "売上予測 ─ 拠点×月25万円。5年目は月2,400万円"), "利用料が毎年積み上がり、導入支援と受託開発は新規拠点の数に比例する");
    const years = ["1年目", "2年目", "3年目", "4年目", "5年目"];
    const gx = X0, gw = 4.45, top = 1.15;
    txt(s, "成長曲線（百万円）", { x: gx, y: top, w: gw, h: 0.26, size: 10.5, bold: true });
    txt(s, [run("━ 売上高", { bold: true }), run("　■ 営業利益", { color: K.gray700 }), run("　┅ 累積営業損益", { color: K.gray700 })], { x: gx + 1.4, y: top, w: gw - 1.4, h: 0.26, size: 8, align: "right", valign: "middle" });
    s.addChart([
      { type: pres.charts.LINE, data: [{ name: "売上高", labels: years, values: [12, 48, 110, 190, 290] }], options: { chartColors: [K.ink], lineSize: 3.5, lineDataSymbol: "circle", lineDataSymbolSize: 6, showValue: true } },
      { type: pres.charts.LINE, data: [{ name: "営業利益", labels: years, values: [-32, -27, -6, 28, 69] }], options: { chartColors: [K.gray700], lineSize: 2, lineDataSymbol: "square", lineDataSymbolSize: 5, showValue: true } },
      { type: pres.charts.LINE, data: [{ name: "累積営業損益", labels: years, values: [-32, -59, -65, -37, 32] }], options: { chartColors: [K.gray500], lineSize: 2, lineDash: "dash", lineDataSymbol: "none", showValue: false } },
    ], { x: gx, y: top + 0.28, w: gw, h: 2.3, catAxisLabelColor: K.gray700, valAxisLabelColor: K.gray700, catAxisLabelFontSize: 9, valAxisLabelFontSize: 8,
      catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", valGridLine: { color: K.rule, size: 0.5 }, catGridLine: { style: "none" },
      valAxisMinVal: -100, valAxisMaxVal: 300, valAxisMajorUnit: 50, valAxisLineShow: false, showLegend: false,
      dataLabelFontSize: 8, dataLabelFontFace: "+mn-lt", dataLabelColor: K.ink, dataLabelPosition: "t", catAxisLabelPos: "low", objectName: "growth-chart" });
    box(s, { x: gx, y: 3.85, w: gw, h: 1.05, fill: K.ink, name: "monthly" });
    txt(s, [
      run("月次売上の根拠（5年目の月平均 約2,420万円）", { bold: true, fontSize: 10, breakLine: true }),
      run("利用料 約1,790万円 ＝ 月25万円×平均約72拠点", { breakLine: true }),
      run("導入支援 約550万円 ＝ 200万円×月2.75拠点（新規33拠点÷12）", { breakLine: true }),
      run("受託開発 約80万円"),
    ], { x: gx + 0.15, y: 3.88, w: gw - 0.3, h: 1.0, size: 8.5, color: K.paper, psa: 1 });

    const rx = 5.05, rw = 4.55;
    txt(s, "費用の内訳（万円）", { x: rx, y: top, w: rw, h: 0.26, size: 10.5, bold: true });
    table(s, [
      ["", "1年目", "3年目", "5年目"],
      ["社員数", "4名", "11名", "20名"],
      ["人件費（販管費分）", "2,480", "5,980", "11,470"],
      ["家賃", "240", "360", "600"],
      ["広告・展示会", "200", "300", "500"],
      ["交通・出張（現場導入）", "360", "600", "1,000"],
      ["端末・ツール", "200", "300", "400"],
      ["専門家（税理士・弁護士・弁理士）", "100", "150", "200"],
      ["雑費", "100", "190", "200"],
      ["販管費 合計", "3,680", "7,880", "14,370"],
    ], { x: rx, y: top + 0.3, w: rw, colW: [2.05, 0.83, 0.83, 0.84], rowH: 0.3, size: 8, strongRows: [9], name: "costs" });
    txt(s, "所在地案：川崎市（準備期は川崎・横浜のインキュベーション施設）。家賃は1年目15坪・月20万円、5年目40坪・月50万円。人件費は1人年800万円", { x: rx, y: 4.6, w: rw, h: 0.4, size: S.note, color: K.gray500 });
    txt(s, "売上の内訳：既存顧客の利用料（前年末拠点×300万円）＋新規の利用料（新規×300万円×0.5）＋導入支援（新規×200万円）＋受託開発", { x: X0, y: 5.04, w: W, h: 0.22, size: S.note, color: K.gray500 });
  }

  // ---------- Slide 10 ⑥損益分岐点 ----------
  {
    const s = add(10);
    heading(s, chap("⑥", "損益分岐点 ─ 4年目に超え、5年目の余裕率は32%"), "損益分岐点売上＝販管費（固定費）÷粗利率");
    const years = ["1年目", "2年目", "3年目", "4年目", "5年目"];
    const gx = X0, gw = 5.2, top = 1.15;
    txt(s, [run("━ 実際の売上", { bold: true }), run("　┅ 損益分岐点売上（百万円）", { color: K.gray700 })], { x: gx, y: top, w: gw, h: 0.26, size: 8.5, valign: "middle" });
    s.addChart([
      { type: pres.charts.LINE, data: [{ name: "売上高", labels: years, values: [12, 48, 110, 190, 290] }], options: { chartColors: [K.ink], lineSize: 3, lineDataSymbol: "circle", lineDataSymbolSize: 6, showValue: true, dataLabelPosition: "b" } },
      { type: pres.charts.LINE, data: [{ name: "損益分岐点売上", labels: years, values: [89, 93, 119, 151, 196] }], options: { chartColors: [K.gray500], lineSize: 2, lineDash: "dash", lineDataSymbol: "diamond", lineDataSymbolSize: 6, showValue: true, dataLabelPosition: "t" } },
    ], { x: gx, y: top + 0.28, w: gw, h: 3.3, catAxisLabelColor: K.gray700, valAxisLabelColor: K.gray700, catAxisLabelFontSize: 9, valAxisLabelFontSize: 8,
      catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", valGridLine: { color: K.rule, size: 0.5 }, catGridLine: { style: "none" },
      valAxisMinVal: 0, valAxisMaxVal: 300, valAxisMajorUnit: 50, valAxisLineShow: false, showLegend: false,
      dataLabelFontSize: 8, dataLabelFontFace: "+mn-lt", dataLabelColor: K.ink, objectName: "bep-chart" });
    const rx = 5.85, rw = 3.75;
    table(s, [
      ["百万円", "3年目", "4年目", "5年目"],
      ["販管費（固定費）", "79", "107", "144"],
      ["粗利率", "66%", "71%", "73%"],
      ["損益分岐点売上", "119", "151", "196"],
      ["実際の売上", "110", "190", "290"],
      ["安全余裕率", "─", "21%", "32%"],
    ], { x: rx, y: top, w: rw, colW: [1.5, 0.75, 0.75, 0.75], rowH: 0.32, size: 8.5, strongRows: [3], name: "bep-table" });
    box(s, { x: rx, y: top + 2.25, w: rw, h: 1.33, fill: K.ink, name: "bep-people" });
    txt(s, [
      run("人で見た損益分岐", { bold: true, fontSize: 10.5, breakLine: true }),
      run("販管費／人 713万円 ÷ 粗利／拠点 255万円", { breakLine: true }),
      run("＝ 社員1人あたり2.8拠点", { bold: true, breakLine: true }),
      run("3年目 2.73（未達）→ 4年目 3.67（超過）"),
    ], { x: rx + 0.15, y: top + 2.3, w: rw - 0.3, h: 1.25, size: 9, color: K.paper, psa: 2 });
    txt(s, "安全余裕率＝（売上−損益分岐点売上）÷売上。1〜2年目の損益分岐点売上は89・93百万円（売上12・48百万円）", { x: X0, y: 4.98, w: W, h: 0.26, size: S.note, color: K.gray500 });
  }

  // ---------- Slide 10 資金計画・黒字倒産対策 ----------
  {
    const s = add(11);
    heading(s, "⑥ 資金計画・CF ─ 黒字倒産させない仕組みを先に置く", "必要資金1.5億円を4つの手段で段階的に調達し、現金残高が3年目の谷でも6,000万円を残す");
    const lx = X0, lw = 4.3, top = 1.15;
    txt(s, "調達の内訳（必要資金1.5億円）", { x: lx, y: top, w: lw, h: 0.26, size: 10.5, bold: true });
    table(s, [
      ["時期", "手段", "金額", "目的"],
      ["準備期", "出資（今回）", "500万円", "試作品・実証"],
      ["1年目 期首", "日本政策金融公庫\n新規開業・スタートアップ支援資金", "1,000万円", "運転資金"],
      ["1年目 年央", "CVC・VC（電線・ロボット）", "1億1,500万円", "人員・開発"],
      ["2年目〜", "公庫 資本性ローン（予備枠）", "2,000万円", "遅延時のみ"],
      [{ text: "合計", bold: true }, "", { text: "1億5,000万円", bold: true }, ""],
    ], { x: lx, y: top + 0.3, w: lw, colW: [0.8, 1.7, 0.95, 0.85], rowH: 0.36, size: 8, leftCols: [1, 3], strongRows: [5], name: "funding" });

    const rx = 4.95, rw = 4.65;
    txt(s, "期末の現金残高（百万円）", { x: rx, y: top, w: rw, h: 0.26, size: 10.5, bold: true });
    txt(s, [run("━ 現金残高", { bold: true }), run("　┅ 最低残高（固定費3か月分）", { color: K.gray700 })], { x: rx + 1.6, y: top, w: rw - 1.6, h: 0.26, size: 8, align: "right", valign: "middle" });
    const lab = ["準備期", "1年目", "2年目", "3年目", "4年目", "5年目"];
    s.addChart([
      { type: pres.charts.LINE, data: [{ name: "現金残高", labels: lab, values: [0, 93, 66, 60, 88, 157] }], options: { chartColors: [K.ink], lineSize: 3, lineDataSymbol: "circle", lineDataSymbolSize: 6, showValue: true } },
      { type: pres.charts.LINE, data: [{ name: "最低残高", labels: lab, values: [20, 20, 20, 20, 20, 20] }], options: { chartColors: [K.gray500], lineSize: 1.5, lineDash: "dash", lineDataSymbol: "none", showValue: false } },
    ], { x: rx, y: top + 0.28, w: rw, h: 2.2, catAxisLabelColor: K.gray700, valAxisLabelColor: K.gray700, catAxisLabelFontSize: 8.5, valAxisLabelFontSize: 8,
      catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", valGridLine: { color: K.rule, size: 0.5 }, catGridLine: { style: "none" },
      valAxisMinVal: 0, valAxisMaxVal: 175, valAxisMajorUnit: 25, valAxisLineShow: false, showLegend: false,
      dataLabelFontSize: 8, dataLabelFontFace: "+mn-lt", dataLabelColor: K.ink, dataLabelPosition: "t", objectName: "cash-chart" });

    box(s, { x: rx, y: top + 2.6, w: rw, h: 1.15, fill: K.ink, name: "anti-bankruptcy" });
    txt(s, "黒字倒産を防ぐ4つのルール", { x: rx + 0.15, y: top + 2.66, w: rw - 0.3, h: 0.26, size: 10.5, bold: true, color: K.paper });
    txt(s, [
      run("① 利用料は年額前払い。導入支援は着手金50%", { breakLine: true }),
      run("② 現金残高が固定費3か月分を下回る前に、追加調達を始める", { breakLine: true }),
      run("③ 月次の資金繰り表で、入金と支払いの時期を12か月先まで管理", { breakLine: true }),
      run("④ 計画が1年遅れても耐えられる備え（0.56億円）を残高に含める"),
    ], { x: rx + 0.15, y: top + 2.94, w: rw - 0.3, h: 0.78, size: 8.5, color: K.paper, psa: 1 });
    txt(s, "現金残高は簡易計算（期首残高＋調達＋営業損益）。税金・減価償却・融資の返済・運転資本の増減は含めない。固定費3か月分＝3年目の販管費79百万円÷4。公庫の制度：融資限度額7,200万円（うち運転資金4,800万円）", { x: X0, y: 4.98, w: W, h: 0.3, size: S.note, color: K.gray500 });
  }

  // ---------- Slide 12 ⑦チーム・実行力 ----------
  {
    const s = add(12);
    heading(s, chap("⑦", "チーム ─ 現場と技術を揃え、顧問とパートナーで補う"), "創業者の経歴・チームメンバーのスキル・外部パートナーと顧問");
    const lx = X0, lw = 3.4, top = 1.15;
    box(s, { x: lx, y: top, w: lw, h: 3.7, fill: K.ink, name: "founder" });
    txt(s, [run("代表取締役 CEO", { fontSize: 9, breakLine: true }), run("志願者（創業者）", { bold: true, fontSize: 14 })], { x: lx + 0.15, y: top + 0.1, w: lw - 0.3, h: 0.6, color: K.paper });
    txt(s, [
      run("電線メーカー 電力事業部門で、66kV／275kV 地中送電線の施工管理5年", { breakLine: true }),
      run("Python・Power BI・FastAPI・Docker・LangChain・RAG を独学し業務に実装", { breakLine: true }),
      run("ニュース要約→音声合成→毎朝配信の仕組みを個人で構築・運用中", { breakLine: true }),
      run("1級電気工事施工管理技士・G検定・大学院修了", { breakLine: true }),
      run("専念する時期：＿＿＿＿（記入）"),
    ], { x: lx + 0.15, y: top + 0.78, w: lw - 0.3, h: 2.85, size: 9.5, color: K.paper, psa: 6 });
    const rx = 3.95, rw = 5.65;
    txt(s, "チームと外部の体制", { x: rx, y: top, w: rw, h: 0.26, size: 10.5, bold: true });
    table(s, [
      ["役割", "求めるスキル", "時期", "状況"],
      ["技術責任者（CTO）", "映像AI・クラウド開発", "1年目", "（記入）"],
      ["生産技術・現場監修", "電線工場の生産技術の経験", "1年目", "（記入）"],
      ["カスタマーサクセス", "施工管理・製造の現場経験", "1年目", "（記入）"],
      ["顧問", "電線業界・生産技術のOB", "準備期", "（記入）"],
      ["専門家", "税理士・弁護士（データ契約）・弁理士", "準備期", "（記入）"],
      ["パートナー", "スマートグラスメーカー・大学（映像AI）", "準備期〜", "（記入）"],
    ], { x: rx, y: top + 0.3, w: rw, colW: [1.45, 2.4, 0.8, 1.0], rowH: 0.36, size: 8.5, leftCols: [1], name: "team" });
    box(s, { x: rx, y: top + 3.0, w: rw, h: 0.7, fill: K.fill, name: "proof" });
    txt(s, [run("実行力の証拠：", { bold: true }), run("音声の自動配信は毎朝稼働中。本日のデモアプリも自作。最初の3か月で試作品を出す")], { x: rx + 0.15, y: top + 3.0, w: rw - 0.3, h: 0.7, size: 9.5, valign: "middle" });
    txt(s, "状況欄は確定・内諾・打診中・募集中のいずれかを記入する", { x: X0, y: 4.98, w: W, h: 0.26, size: S.note, color: K.gray500 });
  }
  // ---------- Slide 9 ライフサイクル ----------
  {
    const s = add(13);
    heading(s, "事業のライフサイクル ─ 成熟期には上場を目指す", "解決策の3段階を、導入期・成長期・成熟期に重ねる");
    const stg = [
      ["導入期", "準備期〜2年目", "記録と報告を自動に", ["電線工場3拠点で有償試験導入", "赤字。準備期は500万円で試作品と実証", "資金：出資・公庫融資・CVC"]],
      ["成長期", "3〜5年目", "照合して品質を守る", ["30→85拠点。施工会社へ展開", "4年目に黒字化、5年目 営業利益率24%", "資金：自己資金＋予備の資本性ローン"]],
      ["成熟期", "6年目〜", "作業をAIが学ぶ（フィジカルAI）", ["素材産業へ拡大、海外へ輸出", "蓄積した作業データでロボットに教える", "上場を目指す（資本市場から調達）"]],
    ];
    const top = 1.2, cw = 2.97, gap = 0.145, ch = 2.7;
    stg.forEach(([ph, when, prod, items], i) => {
      const x = X0 + i * (cw + gap);
      box(s, { x, y: top + (2 - i) * 0.35, w: cw, h: ch - (2 - i) * 0.35 + 0.35, fill: i === 2 ? K.ink : i === 1 ? K.gray700 : K.fill, name: "life-" + i });
      const c = i === 0 ? K.ink : K.paper;
      const y0 = top + (2 - i) * 0.35;
      txt(s, [run(ph, { bold: true, fontSize: 18, breakLine: true }), run(when, { fontSize: 9.5 })], { x: x + 0.15, y: y0 + 0.1, w: cw - 0.3, h: 0.62, color: c });
      txt(s, prod, { x: x + 0.15, y: y0 + 0.75, w: cw - 0.3, h: 0.3, size: 11, bold: true, color: c });
      txt(s, items.map((v, j) => run("・" + v, { breakLine: j < items.length - 1 })), { x: x + 0.15, y: y0 + 1.1, w: cw - 0.3, h: 1.3, size: 9.5, color: c, psa: 4 });
    });
    box(s, { x: X0, y: 4.35, w: W, h: 0.55, fill: K.fill });
    txt(s, [run("国の方針：", { bold: true }), run("「暗黙知が豊富な現場へのバーティカルAIの開発・実装を推進し、輸出も促進する」─ 内閣府「人工知能基本計画」（令和8年7月14日 閣議決定）")], { x: X0 + 0.15, y: 4.35, w: W - 0.3, h: 0.55, size: 9.5, valign: "middle" });
    txt(s, "上場は目標であり、時期・市場は成長期の実績を見て判断する", { x: X0, y: 5.0, w: W, h: 0.24, size: S.note, color: K.gray500 });
  }

  // ---------- Slide 11 希望金額の使い道・リターン ----------
  {
    const s = add(14);
    heading(s, "希望金額 500万円 ─ 6か月以内に最初の1拠点を動かす", "準備期の資金。創業者の給料はゼロで、すべて試作品と実証に使う");
    const lx = X0, lw = 4.4, top = 1.15;
    txt(s, "使い道（万円）", { x: lx, y: top, w: lw, h: 0.26, size: 10.5, bold: true });
    const uses = [["試作品の開発", 250, "クラウド・生成AI・外注エンジニア"], ["端末・機材", 80, "スマートグラス・ヘルメットカメラ 数台"], ["1拠点での実証", 100, "3か月の交通・設置・現場研修"], ["設立・契約・知財", 40, "法人設立、秘密保持・データ利用契約"], ["予備", 30, ""]];
    const k = (lw - 2.5) / 250;
    uses.forEach(([n, v, d], i) => {
      const y = top + 0.32 + i * 0.5;
      txt(s, [run(n, { bold: true, breakLine: true }), run(d, { fontSize: 8, color: K.gray700 })], { x: lx, y, w: 1.85, h: 0.46, size: 9.5, valign: "middle" });
      box(s, { x: lx + 1.9, y: y + 0.1, w: v * k, h: 0.26, fill: i === 4 ? K.gray500 : K.ink });
      txt(s, String(v), { x: lx + 1.9 + v * k + 0.05, y: y + 0.06, w: 0.5, h: 0.34, size: 11, bold: true, valign: "middle" });
    });
    txt(s, "合計 500万円", { x: lx, y: top + 2.85, w: lw, h: 0.3, size: 12, bold: true });

    const rx = 5.05, rw = 4.55;
    txt(s, "マイルストーン", { x: rx, y: top, w: rw, h: 0.26, size: 10.5, bold: true });
    [["3か月", "試作品：一人称の記録 → 写真の自動抽出 → 報告書の自動作成"], ["6か月", "1拠点で実証：報告書の作成時間がどれだけ減ったかを実測"], ["6か月", "有償試験導入3拠点の契約、公庫融資の申込、CVCとの交渉開始"]].forEach(([t, v], i) => {
      const y = top + 0.32 + i * 0.5;
      box(s, { x: rx, y: y + 0.06, w: 0.75, h: 0.34, fill: K.ink });
      txt(s, t, { x: rx, y: y + 0.06, w: 0.75, h: 0.34, size: 9.5, bold: true, color: K.paper, align: "center", valign: "middle" });
      txt(s, v, { x: rx + 0.85, y, w: rw - 0.85, h: 0.46, size: 9.5, valign: "middle" });
    });
    box(s, { x: rx, y: top + 1.95, w: rw, h: 1.25, line: K.ink, lw: T.stroke.strong, name: "return" });
    txt(s, "虎へのリターン（提案）", { x: rx + 0.15, y: top + 2.0, w: rw - 0.3, h: 0.28, size: 10.5, bold: true });
    txt(s, [
      run("条件：", { bold: true }), run("500万円で発行済株式の10%（評価額 出資前4,500万円）", { breakLine: true }),
      run("出口：", { bold: true }), run("成熟期の上場、または事業会社への売却", { breakLine: true }),
      run("お願い：", { bold: true }), run("資金に加え、製造業・建設業の実証先のご紹介"),
    ], { x: rx + 0.15, y: top + 2.3, w: rw - 0.3, h: 0.86, size: 9, psa: 3 });
    txt(s, "出資条件は提案であり、交渉により決める", { x: X0, y: 5.0, w: W, h: 0.24, size: S.note, color: K.gray500 });
  }

  // ---------- Slide 12 リスクと覚悟 ----------
  {
    const s = add(15);
    heading(s, "リスク ─ 最大のリスクは自分。だから役割と判断基準を先に決める", "弱みは隠さない。対策とセットで示す");
    const risks = [
      ["創業者が律速になる", "役割を「型を渡す」に絞る。BSCの4指標で判断を任せ、1か月不在でも回る会社にする"],
      ["製造現場の経験が浅い", "生産技術の経験者を初期メンバーに迎え、顧客の工場に入って学ぶ"],
      ["CADDiなど大手の参入", "図面に現れない現場処置の蓄積と、現場で監修できる人を障壁にする"],
      ["常時収録への抵抗", "作業時間・作業エリアに限定。本人同意と労使合意、休憩中・私語は自動除外"],
      ["資金ショート", "最低残高ルールと前払い・着手金。公庫の資本性ローンを予備枠に持つ"],
    ];
    risks.forEach(([r, a], i) => {
      const y = 1.15 + i * 0.5;
      hline(s, X0, y, W, K.rule, T.stroke.rule);
      txt(s, r, { x: X0, y, w: 2.4, h: 0.5, size: 10.5, bold: true, valign: "middle" });
      txt(s, "→ " + a, { x: X0 + 2.45, y, w: W - 2.45, h: 0.5, size: 9.5, valign: "middle" });
    });
    hline(s, X0, 1.15 + 5 * 0.5, W, K.rule, T.stroke.rule);
    box(s, { x: X0, y: 3.88, w: W, h: 1.2, fill: K.ink, name: "closing" });
    txt(s, "日本の品質を支えてきた現場の勘を、人に依存しない形で残す", { x: X0 + 0.2, y: 3.98, w: W - 0.4, h: 0.4, size: 15, bold: true, color: K.paper });
    txt(s, "その最初の一歩として、500万円で、6か月以内に最初の1拠点を動かします。", { x: X0 + 0.2, y: 4.45, w: W - 0.4, h: 0.5, size: 12, color: K.paper });
  }

  fs.mkdirSync(OUT_DIR, { recursive: true });
  await pres.writeFile({ fileName: OUT });
  console.log("wrote", OUT);
}

main().catch((e) => { console.error(e); process.exit(1); });
