// Build the 令和の虎 pitch deck for GEN³ Works (三現ワークス): 12 slides with speaker notes.
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

function heading(slide, title, lead) {
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
  1: ["0:00 - 0:30", "人が減っても現場が迷わず回る仕組みをつくる。希望金額は500万円",
    "株式会社三現ワークスです。三現とは、現場・現物・現実のことです。工場や電気工事の現場で、熟練者の勘が退職とともに消えています。それを、スマートグラスで記録して、次の担い手とAIに渡す会社をつくります。希望金額は500万円です。よろしくお願いします。"],
  2: ["0:30 - 1:05", "現場を知り、仕組みを自分で作れる人間が、この課題に挑む",
    "私は電線メーカーで、66kVと275kVの地中送電線の施工管理を5年間やってきました。マンホールに入り、ケーブルがつながる瞬間を見てきました。同時に、PythonやAIを独学して、業務の仕組みを自分で作ってきました。毎朝AIがニュースを要約して音声で配信する仕組みも、個人で作って動かしています。現場の言葉と、システムの言葉の両方が分かる。それが私の武器です。"],
  3: ["1:05 - 1:40", "デジタル化は進んだのに、技能継承は人頼みのまま",
    "解きたい課題です。製造業で業務改善にデジタルを使っている企業は77.2%。ところが、それで技能継承が円滑になった企業は8.7%しかありません。技能継承でいちばん多いのは、ベテランに居続けてもらうこと、54.8%です。つまり、まだ人に頼っている。ベテランが辞めた瞬間に、その勘は消えます。"],
  4: ["1:40 - 2:30", "一人称の作業データを撮り続け、報告を自動にし、人とAIの教材にする",
    "解決策です。作業者がスマートグラスやヘルメットカメラを着けるだけで、一人称の動画と会話を撮り続けます。AIがそこから、条件・処置・理由を取り出します。若手には、図面や手順を現物に重ねて違いを知らせる。管理者には、写真を抜き出して当日の報告書を自動で作る。そして後からは教材にして届けます。記録が増えるほど賢くなり、最後はロボットに作業を教えるフィジカルAIにつながります。"],
  5: ["2:30 - 3:20", "デモ：図面との差異を見つけ、報告書が自動でできる",
    "実際の画面をご覧ください。ケーブル接続の現場です。若手が半導電層を処理したところで、図面より8ミリ短いことを検出しました。遠隔の熟練者が、測り直すよう指示します。その会話も記録されます。管理者の画面では、手順ごとに写真が自動で割り当てられ、報告書ができていきます。作業者が報告書に使った時間はゼロです。"],
  6: ["3:20 - 3:55", "顧客は1拠点あたり年300万円を払い、約2か月で元を取る",
    "お金の流れです。お客様は1拠点あたり年300万円、初年度だけ導入支援200万円を加えて500万円です。対象者100名の拠点なら、日報などの書類作業が月6時間減り、毎月240万円分が浮きます。約2.1か月で元が取れます。当社の利用料は粗利率85%で、毎年積み上がります。"],
  7: ["3:55 - 4:30", "「貯める」会社は現れた。当社は「渡す」で分かれる",
    "競合です。正直に申し上げると、記録を貯める会社はすでにあります。東大発のAirion、そしてキャディのCADDiです。ただ、どちらも貯めるところまで。連続工程の現場で、現物に重ねて渡し、教材にまでする会社はまだありません。最大の脅威はCADDiですが、図面に現れない現場の処置は、当社にしか貯まりません。"],
  8: ["4:30 - 5:10", "4年目に損益分岐点を超え、5年目に売上2.9億円",
    "5年計画です。売上は1年目1,200万円から、5年目に2億9,000万円。損益分岐点売上は、販管費を粗利率で割って出します。4年目は1億5,100万円に対して売上1億9,000万円で、ここで初めて分岐点を超えます。5年目の安全余裕率は32%です。"],
  9: ["5:10 - 5:40", "導入期・成長期を越え、成熟期には上場を目指す",
    "事業のライフサイクルです。導入期は記録と報告の自動化で顧客をつかみます。成長期は図面との照合と品質管理で黒字化します。成熟期はフィジカルAIと海外展開に進み、上場を目指します。国の人工知能基本計画も、暗黙知の多い現場へのAIの実装と輸出を推進すると書いています。"],
  10: ["5:40 - 6:20", "黒字倒産しない。最低残高を固定費3か月分に保つ資金計画",
    "資金計画です。必要資金は1.5億円。今回の500万円で準備期を走り、実績を作って日本政策金融公庫の創業融資1,000万円、そして電線メーカーやロボットメーカーのCVCから1億1,500万円を調達します。現金残高は、いちばん薄い3年目でも6,000万円で、固定費3か月分の2,000万円を下回りません。利用料は年額前払い、導入支援は着手金をいただき、黒字倒産を防ぎます。"],
  11: ["6:20 - 6:50", "500万円で、6か月以内に最初の1拠点を動かす",
    "500万円の使い道です。試作品の開発に250万円、スマートグラスなどの端末に80万円、1拠点での実証に100万円、設立と契約に40万円、予備30万円。私の給料はゼロです。3か月で試作品、6か月で実証の結果を出し、有償の試験導入と融資につなげます。"],
  12: ["6:50 - 7:30", "最大のリスクは自分。だから役割と判断基準を先に決めておく",
    "最後にリスクです。最大のリスクは、私自身が律速になることです。だから役割を絞り、判断の基準を先に決めて、私が1か月いなくても回る会社にします。日本の品質を支えてきた現場の勘を、人に依存しない形で残したい。そのための最初の一歩に、500万円をお願いします。"],
};
const NOTES = Object.fromEntries(Object.entries(NOTE_PARTS).map(([n, [t, msg, body]]) => [n, `【${t}】\n要約：${msg}\n\n${body}`]));
const ASSETS = path.join(__dirname, "assets");


async function main() {
  const pres = new pptxgen();
  pres.layout = T.slide.layout;
  pres.title = "株式会社三現ワークス 事業計画書";
  pres.theme = { headFontFace: F, bodyFontFace: F };

  const markBlack = await logo("gen3-mark-black");
  const markWhite = await logo("gen3-mark-white");
  const wordWhite = await logo("gen3-wordmark-white");

  pres.defineSlideMaster({
    title: "GR_CONTENT",
    background: { color: K.paper },
    objects: [
      { image: { x: 9.42, y: 0.22, w: 0.3, h: 0.3, data: markBlack } },
      { text: { text: TAGLINE, options: { x: 6.3, y: 5.3, w: 2.9, h: 0.22, fontFace: F, fontSize: S.footer, color: K.gray500, align: "right", margin: 0, valign: "middle" } } },
    ],
    slideNumber: { x: 9.3, y: 5.3, w: 0.4, h: 0.22, fontFace: F, fontSize: S.footer, color: K.gray500, align: "right", margin: 0 },
  });
  const add = (n) => { const s = pres.addSlide({ masterName: "GR_CONTENT" }); s.addNotes(NOTES[n]); return s; };

  // ---------- Slide 1 表紙・希望金額 ----------
  {
    const s = add(1);
    box(s, { x: 0, y: 0, w: 3.6, h: 5.625, fill: K.ink, name: "cover-panel" });
    s.addImage({ data: wordWhite, x: 0.35, y: 1.1, w: 2.9, h: 0.7 });
    txt(s, "現場の勘を、\n次の担い手へ。", { x: 0.4, y: 2.15, w: 3.0, h: 0.85, size: 20, bold: true, color: K.paper, lsm: 1.2 });
    txt(s, "株式会社三現ワークス（事業計画）", { x: 0.4, y: 3.2, w: 3.0, h: 0.3, size: 11, color: K.paper });
    txt(s, "三現＝現場・現物・現実", { x: 0.4, y: 3.52, w: 3.0, h: 0.28, size: 9.5, color: K.paper });
    const rx = 4.0, rw = 5.3;
    txt(s, "事業計画書", { x: rx, y: 0.55, w: rw, h: 0.28, size: S.lead, color: K.gray700 });
    txt(s, "人が減っても、\n現場が迷わず回る仕組みをつくる", { x: rx, y: 0.9, w: rw, h: 0.85, size: 22, bold: true, lsm: 1.2 });
    txt(s, "スマートグラスで熟練者の作業を記録し、報告を自動にし、次の担い手とAIの教材にする", { x: rx, y: 1.85, w: rw, h: 0.5, size: 11.5 });
    box(s, { x: rx, y: 2.6, w: rw, h: 1.3, line: K.ink, lw: T.stroke.strong, name: "ask" });
    txt(s, "希望金額", { x: rx + 0.2, y: 2.68, w: 2, h: 0.3, size: 12, bold: true, color: K.gray700 });
    txt(s, "500万円", { x: rx + 0.2, y: 2.95, w: 3.2, h: 0.8, size: 44, bold: true, valign: "middle" });
    txt(s, "準備期6か月の資金\n（出資）", { x: rx + 3.45, y: 3.05, w: 1.75, h: 0.6, size: 10, color: K.gray700, valign: "middle" });
    txt(s, "志願者：＿＿＿＿＿＿＿＿＿＿", { x: rx, y: 4.25, w: rw, h: 0.3, size: 10.5, color: K.gray700 });
  }

  // ---------- Slide 2 志願者 ----------
  {
    const s = add(2);
    heading(s, "志願者 ─ 現場を知り、仕組みを自分で作れる", "現場の言葉と、システムの言葉の両方が分かる。それがこの事業の出発点");
    const cards = [
      ["現場を知っている", ["電線メーカーの電力事業部門で、66kV／275kV 地中送電線の施工管理を5年", "マンホールに入り、ケーブル接続・試験に立ち会ってきた", "1級電気工事施工管理技士"]],
      ["仕組みを作れる", ["Python・Power BI・FastAPI・Docker・LangChain・RAG を独学し、業務に実装", "AIがニュースを要約し、音声で毎朝配信する仕組みを個人で構築・運用中", "G検定・大学院修了"]],
      ["この事業をやる理由", ["熟練者が辞めるたびに、現場の判断が消えるのを見てきた", "記録を残す道具はあっても、現場で使われる形で渡す人がいない", "その「訳す役」を、自分が担いたい"]],
    ];
    const cw = 2.97, gap = 0.145, top = 1.15, ch = 3.0;
    cards.forEach(([t, items], i) => {
      const x = X0 + i * (cw + gap);
      box(s, { x, y: top, w: cw, h: ch, fill: i === 2 ? K.ink : K.fill, name: "profile-" + i });
      const c = i === 2 ? K.paper : K.ink;
      txt(s, t, { x: x + 0.15, y: top + 0.12, w: cw - 0.3, h: 0.34, size: 13, bold: true, color: c });
      txt(s, items.map((v, j) => run("・" + v, { breakLine: j < items.length - 1 })), { x: x + 0.15, y: top + 0.55, w: cw - 0.3, h: 2.35, size: 10.5, color: c, psa: 6 });
    });
    txt(s, [run("創業に専念する時期：", { bold: true }), run("＿＿＿＿＿＿（記入）")], { x: X0, y: 4.35, w: W, h: 0.3, size: 11 });
    txt(s, "実物：音声の自動配信は GitHub Actions で毎朝稼働中（ニュース取得→要約→音声合成→ポッドキャスト配信）", { x: X0, y: 4.75, w: W, h: 0.3, size: S.note, color: K.gray500 });
  }

  // ---------- Slide 2 課題（設問①：選んだ理由） ----------
  {
    const s = add(3);
    heading(s, "設問①　課題 ─ デジタル化は進んだのに、技能継承は人頼みのまま", "技能継承の主流は「ベテランに居続けてもらう」。デジタル活用で継承が円滑になった企業は8.7%");
    const lx = X0, lw = 4.6, top = 1.15;
    box(s, { x: lx, y: top, w: lw, h: 1.22, fill: K.fill, name: "example" });
    txt(s, "例：電線の押出ラインで外径が振れたとき", { x: lx + 0.15, y: top + 0.08, w: lw - 0.3, h: 0.28, size: 11, bold: true });
    txt(s, [
      run("熟練者", { bold: true }), run("　音と湿度を見て、押出温度を少し下げて立て直す", { breakLine: true }),
      run("手順書", { bold: true }), run("　この判断は書かれていない", { breakLine: true }),
      run("退職後", { bold: true }), run("　若手は原因が分からず、ラインを止めて不良を出す"),
    ], { x: lx + 0.15, y: top + 0.4, w: lw - 0.3, h: 0.78, size: 10, psa: 3 });
    txt(s, "なぜ自分が取り組むのか", { x: lx, y: top + 1.5, w: lw, h: 0.28, size: 12, bold: true });
    txt(s, [
      run("現場を知っている：", { bold: true }), run("66kV／275kV 地中送電線の施工管理5年。製品が使われる現場の不具合と試験値を実地で把握", { breakLine: true }),
      run("仕組みを作れる：", { bold: true }), run("Python・FastAPI・Docker・LangChain・RAG を独学し業務に実装。音声の自動配信も個人で構築", { breakLine: true }),
      run("資格：", { bold: true }), run("1級電気工事施工管理技士・G検定"),
    ], { x: lx, y: top + 1.8, w: lw, h: 1.5, size: 10, psa: 4 });

    // gap: digital adoption vs skill transfer
    const rx = 5.25, rw = 4.35;
    txt(s, "デジタル化と技能継承のギャップ（製造業）", { x: rx, y: top, w: rw, h: 0.26, size: 10.5, bold: true });
    const bx = rx + 0.05, bw = rw - 0.75;
    [["業務改善にデジタル技術を活用している企業 (1)", 77.2, K.ink], ["その効果として、技能継承が円滑になった企業 (2)", 8.7, K.gray700]].forEach(([lab, v, c], i) => {
      const y = top + 0.34 + i * 0.68;
      txt(s, lab, { x: rx, y, w: rw, h: 0.22, size: 8.5 });
      box(s, { x: bx, y: y + 0.24, w: bw, h: 0.3, fill: K.fill });
      box(s, { x: bx, y: y + 0.24, w: bw * v / 100, h: 0.3, fill: c, name: "bar-" + i });
      txt(s, v + "%", { x: bx + bw * v / 100 + 0.06, y: y + 0.22, w: 0.75, h: 0.34, size: 14, bold: true, valign: "middle" });
    });
    const cy = top + 1.82, cw = (rw - 0.1) / 2, ch = 1.45;
    [
      ["54.8%", "技能継承の取組で最も多いのは「再雇用・勤務延長でベテランに居続けてもらう」(3)"],
      ["62.8%", "製造業の育成上の問題「指導する人材が不足」。「育成する時間がない」も45.4% (3)"],
    ].forEach(([fig, lab], i) => {
      const x = rx + i * (cw + 0.1);
      box(s, { x, y: cy, w: cw, h: ch, line: K.rule, name: "stat-" + i });
      txt(s, fig, { x: x + 0.12, y: cy + 0.1, w: cw - 0.24, h: 0.45, size: 20, bold: true, valign: "middle" });
      txt(s, lab, { x: x + 0.12, y: cy + 0.6, w: cw - 0.24, h: 0.8, size: 9 });
    });
    txt(s, "(1) 労働政策研究・研修機構 調査シリーズNo.267（2026年4月）　(2) 同 No.265（2026年3月）　(3) 経済産業省・厚生労働省・文部科学省「2026年版 ものづくり白書」（令和8年5月）", { x: X0, y: 4.96, w: W, h: 0.3, size: S.note, color: K.gray500 });
  }


  // ---------- Slide 3 解決策（設問①：事業内容）＋R&D ----------
  {
    const s = add(4);
    heading(s, "設問①　解決策 ─ 引退した熟練者が、隣で教えてくれる現場", "一人称の動画と会話を撮り続け、報告は自動に。データは人の教材になり、やがてフィジカルAIの先生になる");
    const steps = [
      ["① 自動で記録", "グラス・カメラ", "作業者の一人称動画と会話を常時収録。記録のための撮影や入力は不要", "市販の端末。構造化を現場に求めない (1)"],
      ["② 整理する", "生成AI", "映像と会話から「条件・処置・理由」を取り出し、ロット・設備データ・図面と紐づける", "生成AI＋現場データ基盤（クラウド）"],
      ["③ 渡す", "若手・管理者", "若手へ：図面・手順を現物に重ね、差異を表示\n管理者へ：写真を抽出し、報告書を自動作成\n後から：教材を当社が制作 (2)", "グラス表示・自動報告・当社の監修"],
    ];
    const top = 1.1, cw = 2.82, ch = 1.42, ag = 0.37;
    steps.forEach(([h, who, what, tool], i) => {
      const x = X0 + i * (cw + ag);
      box(s, { x, y: top, w: cw, h: ch, line: K.rule, name: "step-" + (i + 1) });
      box(s, { x, y: top, w: cw, h: 0.3, fill: K.ink });
      txt(s, [run(h, { bold: true }), run("　" + who, { fontSize: 9 })], { x: x + 0.12, y: top, w: cw - 0.24, h: 0.3, size: 11, color: K.paper, valign: "middle" });
      txt(s, what, { x: x + 0.12, y: top + 0.36, w: cw - 0.24, h: 0.68, size: i === 2 ? 8.5 : 9.5, psa: i === 2 ? 2 : 0 });
      txt(s, [run("道具：", { bold: true }), run(tool)], { x: x + 0.12, y: top + 1.05, w: cw - 0.24, h: 0.34, size: 8.5, color: K.gray700 });
      if (i < 2) txt(s, "▶", { x: x + cw, y: top + ch / 2 - 0.18, w: ag, h: 0.36, size: 14, align: "center", valign: "middle" });
    });

    // three-stage evolution toward physical AI
    const lx = X0, lw = 5.0, by = 2.62;
    txt(s, "3段階で進化する（製品戦略：記録と報告から始め、フィジカルAIへ）", { x: lx, y: by, w: lw, h: 0.24, size: 9.5, bold: true });
    const stages = [
      ["段階1　1〜2年目", "記録と報告を自動に", ["一人称動画と会話を常時収録。写真撮影は不要", "報告書の手順に沿って写真を抽出し、日報を自動作成", "立会できなくても遠隔で確認"], "基盤・KNACK（教材）"],
      ["段階2　3〜4年目", "照合して品質を守る", ["図面・手順と現物を照合し差異を検出", "手順ミスを履歴で後追い確認", "手順ごとの解説動画をAIが生成"], "SAFE・GLASS・TWIN・DRAW"],
      ["段階3　5年目〜", "作業をAIが学ぶ", ["蓄積した一人称動画で作業をAIが学習", "学んだ手順をロボット・検査AIへ", "力加減は触覚シミュレーターで訓練"], "HAPTIC・FLOW（R&D）"],
    ];
    const sw = (lw - 0.2) / 3, sy = by + 0.28, sh = 2.02;
    stages.forEach(([when, ttl, items, mods], i) => {
      const x = lx + i * (sw + 0.1);
      box(s, { x, y: sy, w: sw, h: sh, line: K.rule, fill: i === 2 ? K.fill : K.paper, name: "stage-" + (i + 1) });
      box(s, { x, y: sy, w: sw, h: 0.44, fill: K.ink });
      txt(s, [run(when, { fontSize: 8, breakLine: true }), run(ttl, { bold: true, fontSize: 9.5 })], { x: x + 0.08, y: sy + 0.02, w: sw - 0.16, h: 0.4, color: K.paper, valign: "middle" });
      txt(s, items.map((t, j) => run("・" + t, { breakLine: j < items.length - 1 })), { x: x + 0.08, y: sy + 0.5, w: sw - 0.16, h: 1.12, size: 8, psa: 1 });
      txt(s, mods, { x: x + 0.08, y: sy + sh - 0.38, w: sw - 0.16, h: 0.34, size: 8, bold: true, color: K.gray700 });
    });

    // R&D strategy + onboarding
    const rx = 5.6, rw = 4.0;
    box(s, { x: rx, y: by, w: rw, h: 1.5, fill: K.ink, name: "rnd" });
    txt(s, "R&D戦略：記録ではなく「理解」をつくる", { x: rx + 0.12, y: by + 0.05, w: rw - 0.24, h: 0.26, size: 10.5, bold: true, color: K.paper });
    txt(s, [
      run("記録と理解は違う。どの場面が重要かを見分けられるのは、現場を知る人", { breakLine: true }),
      run("作らない：", { bold: true }), run("映像・音声の記録とAI構造化は汎用技術", { breakLine: true }),
      run("作る：", { bold: true }), run("現場経験者が映像に意味（手順・重要場面）を付け、人のレビューで学習を回し続ける仕組み", { breakLine: true }),
      run("組む：", { bold: true }), run("グラス・ヘルメットカメラ・CAD・3Dはメーカー・大学と"),
    ], { x: rx + 0.12, y: by + 0.32, w: rw - 0.24, h: 1.14, size: 8.5, color: K.paper, psa: 2 });
    box(s, { x: rx, y: by + 1.6, w: rw, h: 0.58, fill: K.fill, name: "onboarding" });
    txt(s, [run("道具以外：導入支援（初年度・約2か月）", { bold: true, fontSize: 9.5, breakLine: true }), run("当社の担当者が現場に入り、聞き取り・帳票の移行・研修まで行う")], { x: rx + 0.12, y: by + 1.6, w: rw - 0.24, h: 0.58, size: 8.5, valign: "middle" });
    txt(s, "(1) 内閣府「人工知能基本計画」（令和8年7月14日 閣議決定）「完全なデータの構造化をしない形での現場データの活用を推進する」　(2) 指導役不足59.5%・育成時間なし47.4%：厚生労働省「令和6年度 能力開発基本調査」（2025年6月公表）。常時収録は作業時間・作業エリアに限り、本人同意のもとで行う", { x: X0, y: 4.94, w: W, h: 0.3, size: S.note, color: K.gray500 });
  }

  // ---------- Slide 5 デモ ----------
  {
    const s = add(5);
    heading(s, "デモ ─ 図面との差異を見つけ、報告書が自動でできる", "架空のケーブル接続工事。若手の作業を記録し、8mmの差異を検出、報告書と教材までつながる");
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

  // ---------- Slide 4 収益の仕組み ----------
  {
    const s = add(6);
    heading(s, "収益の仕組み ─ 顧客は年300万円を払い、約2か月で元を取る", "当社の収入は、毎年積み上がる利用料と、初年度だけの導入支援の二本");
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
    box(s, { x: lx, y: dy + 1.56, w: lw, h: 0.52, fill: K.ink });
    txt(s, [run("当社の利益：", { bold: true }), run("利用料300万円は粗利率85%で毎年積み上がり、導入支援は粗利率40%で一度きり。顧客が増えるほど利用料が積み上がる")], { x: lx + 0.12, y: dy + 1.56, w: lw - 0.24, h: 0.52, size: 9, color: K.paper, valign: "middle" });

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

  // ---------- Slide 5 競合（貯める／渡す） ----------
  {
    const s = add(7);
    heading(s, "競合 ─ 「貯める」会社は現れた。当社は「渡す」で分かれる", "判断の理由を扱う会社は現れた。連続工程の現場で、現物に重ねて渡し、教材にまでする会社はまだない");
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
  // ---------- Slide 8 5年計画・損益分岐点 ----------
  {
    const s = add(8);
    heading(s, "5年計画 ─ 4年目に損益分岐点を超え、5年目に売上2.9億円", "損益分岐点売上＝販管費（固定費）÷粗利率。4年目に実際の売上が初めて上回る");
    const years = ["1年目", "2年目", "3年目", "4年目", "5年目"];
    const axis = (min, max, unit) => ({ catAxisLabelColor: K.gray700, valAxisLabelColor: K.gray700, catAxisLabelFontSize: 9, valAxisLabelFontSize: 8,
      catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", valGridLine: { color: K.rule, size: 0.5 }, catGridLine: { style: "none" },
      valAxisMinVal: min, valAxisMaxVal: max, valAxisMajorUnit: unit, valAxisLineShow: false, showLegend: false,
      dataLabelFontSize: 8, dataLabelFontFace: "+mn-lt", dataLabelColor: K.ink, dataLabelPosition: "t", catAxisLabelPos: "low" });
    const gx = X0, gw = 4.55, top = 1.15;
    txt(s, "成長曲線（百万円）", { x: gx, y: top, w: gw, h: 0.26, size: 10.5, bold: true });
    txt(s, [run("━ 売上高", { bold: true }), run("　■ 営業利益", { color: K.gray700 }), run("　┅ 累積営業損益", { color: K.gray700 })], { x: gx + 1.4, y: top, w: gw - 1.4, h: 0.26, size: 8, align: "right", valign: "middle" });
    s.addChart([
      { type: pres.charts.LINE, data: [{ name: "売上高", labels: years, values: [12, 48, 110, 190, 290] }], options: { chartColors: [K.ink], lineSize: 3.5, lineDataSymbol: "circle", lineDataSymbolSize: 6, showValue: true } },
      { type: pres.charts.LINE, data: [{ name: "営業利益", labels: years, values: [-32, -27, -6, 28, 69] }], options: { chartColors: [K.gray700], lineSize: 2, lineDataSymbol: "square", lineDataSymbolSize: 5, showValue: true } },
      { type: pres.charts.LINE, data: [{ name: "累積営業損益", labels: years, values: [-32, -59, -65, -37, 32] }], options: { chartColors: [K.gray500], lineSize: 2, lineDash: "dash", lineDataSymbol: "none", showValue: false } },
    ], { x: gx, y: top + 0.28, w: gw, h: 2.5, ...axis(-100, 300, 50), objectName: "growth-chart" });

    const rx = 5.15, rw = 4.45;
    txt(s, "損益分岐点（百万円）", { x: rx, y: top, w: rw, h: 0.26, size: 10.5, bold: true });
    txt(s, [run("━ 実際の売上", { bold: true }), run("　┅ 損益分岐点売上", { color: K.gray700 })], { x: rx + 1.4, y: top, w: rw - 1.4, h: 0.26, size: 8, align: "right", valign: "middle" });
    s.addChart([
      { type: pres.charts.LINE, data: [{ name: "売上高", labels: years, values: [12, 48, 110, 190, 290] }], options: { chartColors: [K.ink], lineSize: 3, lineDataSymbol: "circle", lineDataSymbolSize: 6, showValue: true, dataLabelPosition: "b" } },
      { type: pres.charts.LINE, data: [{ name: "損益分岐点売上", labels: years, values: [89, 93, 119, 151, 196] }], options: { chartColors: [K.gray500], lineSize: 2, lineDash: "dash", lineDataSymbol: "diamond", lineDataSymbolSize: 6, showValue: true, dataLabelPosition: "t" } },
    ], { x: rx, y: top + 0.28, w: rw, h: 2.5, ...axis(0, 300, 50), objectName: "bep-chart" });

    table(s, [
      ["百万円", "1年目", "2年目", "3年目", "4年目", "5年目"],
      ["販管費（固定費）", "37", "56", "79", "107", "144"],
      ["粗利率", "40%", "61%", "66%", "71%", "73%"],
      ["損益分岐点売上", "89", "93", "119", "151", "196"],
      ["安全余裕率", "─", "─", "─", "21%", "32%"],
    ], { x: X0, y: 3.98, w: 6.0, colW: [1.5, 0.9, 0.9, 0.9, 0.9, 0.9], rowH: 0.15, size: 8, strongRows: [3], name: "bep-table" });
    box(s, { x: 6.55, y: 3.98, w: 3.05, h: 0.9, fill: K.ink });
    txt(s, [run("人で見た損益分岐：", { bold: true }), run("社員1人あたり2.8拠点。3年目2.73で届かず、4年目3.67で超える")], { x: 6.67, y: 3.98, w: 2.85, h: 0.9, size: 8.5, color: K.paper, valign: "middle" });
    txt(s, "損益分岐点売上＝販管費÷粗利率（売上総利益÷売上高）。安全余裕率＝（売上−損益分岐点売上）÷売上", { x: X0, y: 5.08, w: W, h: 0.2, size: S.note, color: K.gray500 });
  }

  // ---------- Slide 9 ライフサイクル ----------
  {
    const s = add(9);
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

  // ---------- Slide 10 資金計画・黒字倒産対策 ----------
  {
    const s = add(10);
    heading(s, "資金計画 ─ 黒字倒産させない。最低残高は固定費3か月分", "必要資金1.5億円を4つの手段で段階的に調達し、現金残高が3年目の谷でも6,000万円を残す");
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

  // ---------- Slide 11 希望金額の使い道・リターン ----------
  {
    const s = add(11);
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
    const s = add(12);
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
