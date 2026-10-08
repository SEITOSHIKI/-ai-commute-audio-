// Build the GEN³ Works (三現ワークス) business-plan deck (8 slides, speaker notes) with pptxgenjs.
// Usage: node build_deck.js [outDir]   (needs pptxgenjs and sharp)
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const sharp = require("sharp");
const T = require("../design-system/georec/pptx-theme.js");

const { color: K, size: S } = T;
const OUT_DIR = process.argv[2] || __dirname;
const OUT = path.join(OUT_DIR, "セルフマネジメント課題_事業構想プレゼン.pptx");
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
  1: ["0:00 - 0:50", "人が減っても現場が迷わず回る仕組みをつくり、日本の品質を支える現場の勘を遺産として残す（設問③）",
    "株式会社三現ワークスの事業構想を発表します。三現とは、現場・現物・現実のことです。メイド・イン・ジャパンの品質を支えてきたのは、現場の熟練者の勘です。その勘は、記録に残らないまま、退職とともに失われつつあります。この会社の存在意義は「現場の勘を、次の担い手へ。」です。使命は、熟練者の勘を言葉にし、教材と仕組みで次の人へ渡すこと。5年後には85の拠点で、勘どころが教材として残っている状態を目指します。今年7月に閣議決定された国の人工知能基本計画も、暗黙知が豊富な現場へのAIの実装を推進すると明記しています。この会社は、その現場の側に立ちます。"],
  2: ["0:50 - 2:00", "デジタル化は進んだのに、技能継承は人頼みのまま",
    "まず、解きたい課題です。たとえば押出ラインで外径が振れたとき、熟練者は音や湿度を見て温度を少し下げて立て直します。この判断は手順書に書かれていません。国の調査を見ると、製造業で業務改善にデジタル技術を使っている企業は77.2%にのぼります。ところが、その効果として技能継承が円滑になった企業は8.7%しかありません。デジタル化は進んだのに、技能継承にはほとんど効いていないのです。実際、技能継承の取り組みで最も多いのは、再雇用や勤務延長でベテランに居続けてもらうことで、54.8%です。継承は、いまも人に頼ったままです。さらに製造業の62.8%が、指導する人材が足りないと答えています。私は施工管理として製品が使われる現場を見てきました。そして、仕組みを自分で作れます。だからこの課題に取り組みます。"],
  3: ["2:00 - 3:30", "引退した熟練者が隣で教えてくれる現場をつくる。差別化は、教材にして届けること",
    "設問①、解決策です。目指すのは、引退した熟練者が隣で教えてくれるような現場です。熟練者は、異常に対処したらタブレットに音声で一言残すだけです。構造化を現場に求めない設計で、これは国の人工知能基本計画の方針とも一致します。生成AIがそれを条件・処置・理由に分けて事例にします。ただ、記録と整理はすでに他社も手がけており、当社の差別化にはなりません。差別化は三つ目です。指導役が足りない事業所は59.5%、育成の時間がない事業所は47.4%あります。貯めたデータから教材を作る作業が現場に残ると、使われません。だから当社が、連続工程の勘どころを研修資料、通勤中に聞ける音声番組、短い動画にして届けます。教材化は初年度から提供し、その後、危険箇所の表示、3Dの仮想工場、音声での作図と渡し方を増やします。技術は作らない部分を決めています。音声記録やAIの構造化は汎用技術を使い、自前開発は工程データとの紐づけと教材化の手順に集中します。"],
  4: ["3:30 - 4:40", "顧客は1拠点あたり年300万円を払い、約2か月で元を取る",
    "次に、お金の流れです。お客様の工場1拠点が毎年払うのは、アプリと基盤の利用料100万円、記録するラインの数に応じた登録料100万円、そして教材制作や危険表示など選んだ機能の料金100万円、合わせて年300万円です。初年度だけ、当社の担当者が現場に入る導入支援200万円が加わり、500万円になります。お客様から見ると、右のグラフのとおり、対象者100名の拠点で毎月240万円分の工数が浮き、約2.1か月で500万円を上回ります。ただし月6時間の削減は、建設業の書類電子化の実績を当てはめた仮定です。当社から見ると、利用料は粗利率85%で毎年積み上がり、導入支援は粗利率40%で一度きりです。お客様が増えるほど、利用料が積み上がっていく構造です。"],
  5: ["4:40 - 5:50", "「貯める」会社は現れた。当社は「渡す」で分かれる",
    "最初に自分で自分に突きつけた問いが三つありました。コンサルと何が違うのか、同じことをしている会社はないのか、大手が降りてきたらどうするのか、です。調べると、判断の理由を扱う会社はすでに現れています。東大発のAirionは、ベテランとAIの対話で知識を構造化します。キャディのCADDiは、設計意図や判断履歴をデータ化しています。正直に言えば、貯める部分は当社とほぼ同じ設計です。これは、この課題に市場があることの証明だと考えています。分かれるのは渡す部分です。どの会社も、貯めたものを教材にして届けるところまではやっていません。また、CADDiの対象は組立の製造業で、扱うのは図面とBOMです。電線のような連続工程の現場処置は、図面には現れません。最大の脅威は、CADDiが連続工程に降りてくることです。防御線は、図面に現れない現場処置の蓄積と、現場で監修できる人です。"],
  6: ["5:50 - 6:50", "顧客数は、導入担当の人数で決まる",
    "設問②、組織と採用です。先ほどの導入支援は人の作業なので、顧客を増やせる速さは導入担当の人数で決まります。1拠点に約2か月かかり、並行は2拠点までなので、1名あたり年6拠点が上限です。計画はその6割から8割に置き、1年目は製品が未完成のため3拠点に限定します。累計の顧客拠点は5年目に85、社員は4名から20名とします。採用は、製造や施工の現場経験者にITを教える経路を軸にし、製造知見の不足を補うため、生産技術の経験者を初期メンバーに迎えます。私は代表取締役CEOとして、現場の読み解き方を型にして渡す役割に絞り、技術の責任者は兼任しません。"],
  7: ["6:50 - 8:30", "3年目の谷を1.5億円で越え、4年目に黒字化する",
    "設問④、経営計画です。グラフの横軸が年、縦軸が金額です。売上高は1年目1,200万円から、5年目に2億9,000万円まで伸びます。営業利益は3年目まで赤字で、4年目に2,800万円の黒字に転じます。点線の累積営業損益は、3年目に6,500万円の赤字で底を打ちます。この谷を越えるための資金が1.5億円で、谷の0.65億円、運転資金0.29億円、1年遅れた場合の備え0.56億円を積み上げた額です。調達先には、電線メーカーやロボットメーカーのコーポレートベンチャーキャピタルを想定しています。資金と一緒に、実証の現場と販路を得られるためです。なぜ4年目なのか。右のグラフのとおり、社員1人あたりの顧客拠点が2.8を超えると黒字になります。3年目は2.73で届かず、4年目に3.67で超えます。積み上がる利用料が、人件費を追い越す年です。"],
  8: ["8:30 - 9:50", "最大のリスクは自分。判断基準を先に決めておく",
    "設問⑤、経営課題です。最大のリスクは競合ではなく、創業者である自分が律速になることです。準備期は、試験導入を3拠点に限定し、生産技術の経験者を迎えます。中期は、電線業界が353事業所と小さいため、製品の出口である施工会社へ同じ基盤を広げます。後期は、大手のデータ基盤が連続工程に降りてくることへの備えと、私がいなくても判断できる会社にすることです。そのための仕組みが下段のバランスト・スコアカードで、四つの指標を先に決めておきます。最後に、この事業の顧客・商品・収益は、すべて現在の勤務先の中にも存在します。だから復職後は、生産技術と情報システムの領域で、現場と技術のあいだを訳す役割を担いたいと考えています。そして、自らの役割を定義しきること自体が、業務を際限なく抱え込まないための再発防止策でもあります。以上です。"],
};
const NOTES = Object.fromEntries(Object.entries(NOTE_PARTS).map(([n, [t, msg, body]]) => [n, `【${t}】\n要約：${msg}\n\n${body}`]));

async function main() {
  const pres = new pptxgen();
  pres.layout = T.slide.layout;
  pres.title = "株式会社三現ワークス 事業構想";
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

  // ---------- Slide 1 表紙・Purpose（設問③） ----------
  {
    const s = add(1);
    box(s, { x: 0, y: 0, w: 3.33, h: 5.625, fill: K.ink, name: "cover-panel" });
    s.addImage({ data: wordWhite, x: 0.3, y: 1.2, w: 2.75, h: 0.66 });
    txt(s, "PURPOSE　存在意義", { x: 0.35, y: 2.15, w: 2.9, h: 0.24, size: 9, bold: true, color: K.paper });
    txt(s, "現場の勘を、\n次の担い手へ。", { x: 0.35, y: 2.4, w: 2.9, h: 0.8, size: 20, bold: true, color: K.paper, lsm: 1.2 });
    txt(s, "日本の品質を支える現場の勘を、人に依存しない遺産として残す", { x: 0.35, y: 3.25, w: 2.75, h: 0.5, size: 9.5, color: K.paper });
    txt(s, "株式会社三現ワークス（事業構想）", { x: 0.35, y: 3.85, w: 2.9, h: 0.3, size: 11, color: K.paper });
    txt(s, "三現＝現場・現物・現実", { x: 0.35, y: 4.17, w: 2.9, h: 0.28, size: 9.5, color: K.paper });

    const rx = 3.75, rw = 5.55;
    txt(s, "リワーク実習　セルフマネジメント課題", { x: rx, y: 0.5, w: rw, h: 0.28, size: S.lead, color: K.gray700 });
    txt(s, "人が減っても、\n現場が迷わず回る仕組みをつくる", { x: rx, y: 0.85, w: rw, h: 0.85, size: 22, bold: true, lsm: 1.2 });
    txt(s, "設問③　経営理念", { x: rx, y: 1.86, w: rw, h: 0.24, size: 9.5, bold: true, color: K.gray700 });
    const rows = [
      ["MISSION", "使命", "熟練者の勘を言語化し、教材と仕組みで次の人へ渡す"],
      ["VISION", "5年後", "85拠点で、勘どころが教材として残っている"],
      ["VALUES", "行動指針", "現場に立つ／現物に触れる／現実で判断する"],
    ];
    const y0 = 2.14, rh = 0.5;
    rows.forEach(([en, jp, v], i) => {
      const y = y0 + i * rh;
      hline(s, rx, y, rw, K.rule, T.stroke.rule);
      txt(s, [run(en, { bold: true, fontSize: 10, breakLine: true }), run(jp, { fontSize: 8, color: K.gray700 })], { x: rx, y: y + 0.03, w: 1.0, h: rh - 0.06, valign: "middle" });
      txt(s, v, { x: rx + 1.05, y, w: rw - 1.05, h: rh, size: 11.5, valign: "middle" });
    });
    hline(s, rx, y0 + 3 * rh, rw, K.rule, T.stroke.rule);
    box(s, { x: rx, y: 3.78, w: rw, h: 0.72, fill: K.fill, name: "policy-quote" });
    txt(s, [run("「暗黙知が豊富な現場へのバーティカルAIの開発・実装を推進し、輸出も促進する」", { bold: true, breakLine: true }), run("─ 内閣府「人工知能基本計画 ～日本AX、より強く、より豊かに～」（令和8年7月14日 閣議決定）", { fontSize: 8, color: K.gray700 })], { x: rx + 0.12, y: 3.78, w: rw - 0.24, h: 0.72, size: 9.5, valign: "middle", psa: 2 });
    txt(s, "発表者：＿＿＿＿＿＿＿＿＿＿　　発表時間：10分", { x: rx, y: 4.65, w: rw, h: 0.3, size: 10.5, color: K.gray700 });
  }

  // ---------- Slide 2 課題（設問①：選んだ理由） ----------
  {
    const s = add(2);
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
    const s = add(3);
    heading(s, "設問①　解決策 ─ 引退した熟練者が、隣で教えてくれる現場", "記録と整理は手段。差別化は③：連続工程の勘どころを、研修資料・音声番組・動画にして当社が届ける");
    const steps = [
      ["① 記録する", "熟練者", "異常に対処したら、タブレットに音声で一言残す。例「湿度が高いので温度を2℃下げた」", "記録アプリ（音声）。構造化を現場に求めない (1)"],
      ["② 整理する", "生成AI", "一言を「条件・処置・理由」に分けて文章化し、ロット・設備データと紐づけて事例にする", "生成AI＋現場データ基盤（クラウド）"],
      ["③ 教材にして渡す", "当社", "指導役が足りず（59.5%）、育成の時間もない（47.4%）(2)。だから教材は当社が作って届ける", "生成AIで台本→当社が監修→配信"],
    ];
    const top = 1.12, cw = 2.82, ch = 1.5, ag = 0.37;
    steps.forEach(([h, who, what, tool], i) => {
      const x = X0 + i * (cw + ag);
      box(s, { x, y: top, w: cw, h: ch, line: K.rule, name: "step-" + (i + 1) });
      box(s, { x, y: top, w: cw, h: 0.3, fill: K.ink });
      txt(s, [run(h, { bold: true }), run("　" + who, { fontSize: 9 })], { x: x + 0.12, y: top, w: cw - 0.24, h: 0.3, size: 11, color: K.paper, valign: "middle" });
      txt(s, what, { x: x + 0.12, y: top + 0.36, w: cw - 0.24, h: 0.66, size: 9.5 });
      txt(s, [run("道具：", { bold: true }), run(tool)], { x: x + 0.12, y: top + 1.1, w: cw - 0.24, h: 0.34, size: 8.5, color: K.gray700 });
      if (i < 2) txt(s, "▶", { x: x + cw, y: top + ch / 2 - 0.18, w: ag, h: 0.36, size: 14, align: "center", valign: "middle" });
    });

    // hand-off roadmap: products and R&D
    const lx = X0, lw = 5.0, by = 2.78;
    txt(s, "渡し方を増やす（製品戦略：教材化を初年度から、追加機能は年1本ずつ）", { x: lx, y: by, w: lw, h: 0.24, size: 9.5, bold: true });
    const rows = [
      ["1年目", "基盤", "画面", "記録・過去事例の検索（全顧客）", "製品"],
      ["1年目", "KNACK", "教材", "勘どころの音声番組・動画・研修資料を制作", "製品"],
      ["2年目", "SAFE", "警告", "手順書を開くと過去の事故・ヒヤリ地点を表示", "製品"],
      ["3年目", "TWIN", "3D", "3D仮想工場でレイアウト・ラインの流れを試算", "製品"],
      ["4年目", "DRAW", "図面", "登録したパターンを音声指示でCAD作図", "製品"],
      ["研究", "HAPTIC", "触覚", "力加減を体験できる訓練シミュレーター", "R&D"],
      ["研究", "FLOW", "機械", "搬送ロボット・検査AIへ条件を直接渡す", "R&D"],
    ];
    const rh = 0.25, ry = by + 0.27;
    rows.forEach(([yr, name, form, d, st], i) => {
      const y = ry + i * rh;
      if (st === "R&D") box(s, { x: lx, y, w: lw, h: rh, fill: K.fill });
      hline(s, lx, y, lw, K.rule, T.stroke.rule);
      txt(s, yr, { x: lx + 0.04, y, w: 0.46, h: rh, size: 8, color: K.gray700, valign: "middle" });
      txt(s, name, { x: lx + 0.5, y, w: 0.68, h: rh, size: 8.5, bold: true, valign: "middle" });
      txt(s, form, { x: lx + 1.18, y, w: 0.66, h: rh, size: 8, color: K.gray700, valign: "middle" });
      txt(s, d, { x: lx + 1.84, y, w: lw - 2.3, h: rh, size: 8, valign: "middle" });
      txt(s, st, { x: lx + lw - 0.46, y, w: 0.42, h: rh, size: 8, bold: st === "R&D", align: "right", valign: "middle" });
    });
    hline(s, lx, ry + rows.length * rh, lw, K.rule, T.stroke.rule);

    // R&D strategy + onboarding
    const rx = 5.6, rw = 4.0;
    box(s, { x: rx, y: by, w: rw, h: 1.32, fill: K.ink, name: "rnd" });
    txt(s, "R&D戦略（技術開発）", { x: rx + 0.12, y: by + 0.05, w: rw - 0.24, h: 0.26, size: 10.5, bold: true, color: K.paper });
    txt(s, [
      run("作らない：", { bold: true }), run("音声記録・AI構造化は汎用技術。自前開発は最小限", { breakLine: true }),
      run("作る：", { bold: true }), run("連続工程の工程データとの紐づけ、教材化の手順", { breakLine: true }),
      run("組む：", { bold: true }), run("CAD・3D・触覚デバイスはベンダー・大学と共同研究", { breakLine: true }),
      
      run("試作済み：", { bold: true }), run("ニュース要約→音声合成→ポッドキャスト自動配信を個人で実装"),
    ], { x: rx + 0.12, y: by + 0.32, w: rw - 0.24, h: 0.98, size: 8.5, color: K.paper, psa: 1 });
    box(s, { x: rx, y: by + 1.42, w: rw, h: 0.72, fill: K.fill, name: "onboarding" });
    txt(s, "道具以外：導入支援（初年度・約2か月）", { x: rx + 0.12, y: by + 1.46, w: rw - 0.24, h: 0.26, size: 10, bold: true });
    txt(s, "当社の担当者が現場に入り、熟練者への聞き取り・帳票の移行・現場研修まで行う", { x: rx + 0.12, y: by + 1.72, w: rw - 0.24, h: 0.4, size: 8.5 });
    txt(s, "(1) 内閣府「人工知能基本計画」（令和8年7月14日 閣議決定）「完全なデータの構造化をしない形での現場データの活用を推進する」　(2) 厚生労働省「令和6年度 能力開発基本調査」（2025年6月公表）。R&D行は製品化前の売上を計画に含めない", { x: X0, y: 4.94, w: W, h: 0.3, size: S.note, color: K.gray500 });
  }

  // ---------- Slide 4 収益の仕組み ----------
  {
    const s = add(4);
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
    txt(s, "横軸は経過月数。時間単価4,000円（780万円÷1,950h）×月6h×100名＝月240万円。月6hは建設業の書類電子化実績を準用した仮定。年300万円は顧客効果の約1割（価値基準価格）", { x: rx, y: top + 2.94, w: rw, h: 0.5, size: S.note, color: K.gray700 });
  }

  // ---------- Slide 5 競合（貯める／渡す） ----------
  {
    const s = add(5);
    heading(s, "競合 ─ 「貯める」会社は現れた。当社は「渡す」で分かれる", "判断の理由を扱う会社は現れた。連続工程の現場処置を、教材にして届ける会社はまだない");
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
      [{ text: "三現ワークス", bold: true }, { text: "○ 音声で一言", bold: true }, { text: "○ 当社が制作", bold: true }, { text: "連続工程", bold: true }],
    ];
    table(s, [head, ...rows], { x: sx, y: my + 0.3, w: sw, colW: cols, rowH: 0.27, size: 8, align: "left", strongRows: [6], name: "store-handoff" });
    const ty = my + 0.3 + 7 * 0.27 + 0.1;
    box(s, { x: sx, y: ty, w: sw, h: 0.66, fill: K.ink, name: "threat" });
    txt(s, [
      run("最大の脅威：", { bold: true }), run("CADDiが連続工程に降りてくること。課題認識は同じ（「製造業の叡智の8割以上は暗黙知」）", { breakLine: true }),
      run("防御線：", { bold: true }), run("図面に現れない現場処置の蓄積と、現場で監修できる人"),
    ], { x: sx + 0.12, y: ty, w: sw - 0.24, h: 0.66, size: 8.5, color: K.paper, valign: "middle", psa: 2 });
    txt(s, [
      run("STP：", { bold: true }), run("工程の形（連続／組立／施工）×規模で区分し、中堅の電線工場を狙う", { breakLine: true }),
      run("競争戦略：", { bold: true }), run("ニッチャー（コトラー）×差別化集中（ポーター）。大手の内製は個別開発で中堅に届かない"),
    ], { x: sx, y: ty + 0.74, w: sw, h: 0.5, size: 8.5, psa: 2 });
    txt(s, "出典：各社公表資料（Airion 2025年7月、キャディ 2026年8月6日、三菱総合研究所、FRONTEO、tebiki、スタディスト）。詳細と大手の内製事例は補足資料", { x: X0, y: 4.98, w: W, h: 0.28, size: S.note, color: K.gray500 });
  }
  // ---------- Slide 6 組織と採用 ----------
  {
    const s = add(6);
    heading(s, "設問②　組織 ─ 顧客数は、導入担当の人数で決まる", "導入担当1名あたり年6拠点。代表（CEO）は型を渡す役割に絞り、抱え込まない");
    const lx = X0, lw = 4.5, top = 1.15;
    box(s, { x: lx, y: top, w: lw, h: 1.0, line: K.ink, lw: T.stroke.strong });
    txt(s, "設計基準　導入担当1名あたり 年6拠点", { x: lx + 0.15, y: top + 0.08, w: lw - 0.3, h: 0.32, size: 13, bold: true });
    txt(s, "1拠点の導入＝帳票・試験記録の作り込み・過去データ移行・現場研修で約2か月。並行2拠点が上限 → 12か月÷2か月×1拠点＝年6拠点。計画値は上限の6〜8割", { x: lx + 0.15, y: top + 0.42, w: lw - 0.3, h: 0.54, size: 9.5 });
    table(s, [
      ["", "1年目", "2年目", "3年目", "4年目", "5年目"],
      ["導入担当（名）", "1.5", "2.5", "3.5", "5.5", "7.0"],
      ["理論上限（×6拠点）", "9", "15", "21", "33", "42"],
      ["新規獲得（計画）", "3", "9", "18", "27", "33"],
      ["解約", "0", "0", "0", "2", "3"],
      ["累計顧客拠点数", "3", "12", "30", "55", "85"],
    ], { x: lx, y: top + 1.15, w: lw, colW: [1.5, 0.6, 0.6, 0.6, 0.6, 0.6], rowH: 0.27, strongRows: [5], name: "customers" });
    txt(s, "※1年目は製品が未完成のため、上限9拠点に対し3拠点に限定する（能力ではなく意図的な制約）", { x: lx, y: top + 2.85, w: lw, h: 0.3, size: S.note, color: K.gray700 });

    const rx = 5.1, rw = 4.5;
    table(s, [
      ["人員計画（名）", "1年目", "2年目", "3年目", "4年目", "5年目"],
      ["代表", "1", "1", "1", "1", "1"],
      ["エンジニア", "2", "3", "5", "7", "9"],
      ["カスタマーサクセス", "1", "2", "2", "4", "5"],
      ["営業", "0", "0", "1", "1", "2"],
      ["管理（業務委託含む）", "0", "1", "2", "2", "3"],
      ["合計", "4", "7", "11", "15", "20"],
    ], { x: rx, y: top, w: rw, colW: [1.5, 0.6, 0.6, 0.6, 0.6, 0.6], rowH: 0.27, strongRows: [6], name: "staffing" });
    txt(s, "採用基準と代表（CEO）の役割", { x: rx, y: top + 2.08, w: rw, h: 0.28, size: 12, bold: true });
    txt(s, [
      run("採用の軸：", { bold: true }), run("製造・施工の現場経験者にITを教える経路", { breakLine: true }),
      run("初期メンバー：", { bold: true }), run("生産技術経験者を迎え、製造知見の不足を補う", { breakLine: true }),
      run("選考基準：", { bold: true }), run("現場に出ることを厭わない／「作らない」判断ができる", { breakLine: true }),
      run("代表（CEO）：", { bold: true }), run("現場の読み解き方を型にして渡す（CTO兼任なし）", { breakLine: true }),
      run("5年目の到達目標：", { bold: true }), run("創業者が1か月不在でも事業が回る状態"),
    ], { x: rx, y: top + 2.4, w: rw, h: 1.7, size: 10, psa: 3 });
  }

  // ---------- Slide 7 経営計画（設問④） ----------
  {
    const s = add(7);
    heading(s, "設問④　経営計画 ─ 3年目の谷を越え、4年目に黒字化する", "売上は5年で2.9億円へ。累積損益の谷 0.65億円を、必要資金 1.5億円で越える");
    const years = ["1年目", "2年目", "3年目", "4年目", "5年目"];
    const axis = (min, max, unit) => ({ catAxisLabelColor: K.gray700, valAxisLabelColor: K.gray700, catAxisLabelFontSize: 9, valAxisLabelFontSize: 8,
      catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", valGridLine: { color: K.rule, size: 0.5 }, catGridLine: { style: "none" },
      valAxisMinVal: min, valAxisMaxVal: max, valAxisMajorUnit: unit, valAxisLineShow: false, showLegend: false,
      dataLabelFontSize: 8, dataLabelFontFace: "+mn-lt", dataLabelColor: K.ink, dataLabelPosition: "t", catAxisLabelPos: "low" });
    const gx = X0, gw = 5.65, top = 1.15;
    txt(s, "成長曲線（単位：百万円）", { x: gx, y: top, w: gw, h: 0.26, size: 10.5, bold: true });
    s.addChart([
      { type: pres.charts.LINE, data: [{ name: "売上高", labels: years, values: [12, 48, 110, 190, 290] }], options: { chartColors: [K.ink], lineSize: 3.5, lineDataSymbol: "circle", lineDataSymbolSize: 6, showValue: true } },
      { type: pres.charts.LINE, data: [{ name: "営業利益", labels: years, values: [-32, -27, -6, 28, 69] }], options: { chartColors: [K.gray700], lineSize: 2, lineDataSymbol: "square", lineDataSymbolSize: 5, showValue: true } },
      { type: pres.charts.LINE, data: [{ name: "累積営業損益", labels: years, values: [-32, -59, -65, -37, 32] }], options: { chartColors: [K.gray500], lineSize: 2, lineDash: "dash", lineDataSymbol: "none", showValue: false } },
    ], { x: gx, y: top + 0.28, w: gw, h: 3.05, ...axis(-100, 300, 50), objectName: "growth-chart" });
    // direct labels (no legend: series are told apart by weight, marker and dash)
    txt(s, [run("━ 売上高", { bold: true }), run("　■ 営業利益", { color: K.gray700 }), run("　┅ 累積営業損益（谷 ▲65）", { color: K.gray700 })], { x: gx + 1.8, y: top, w: gw - 1.8, h: 0.26, size: 8.5, align: "right", valign: "middle" });

    const rx = 6.25, rw = 3.35;
    txt(s, "黒字化の条件：社員1人あたり顧客拠点数", { x: rx, y: top, w: rw, h: 0.26, size: 10.5, bold: true });
    s.addChart([
      { type: pres.charts.LINE, data: [{ name: "顧客拠点／社員", labels: years, values: [0.75, 1.71, 2.73, 3.67, 4.25] }], options: { chartColors: [K.ink], lineSize: 3, lineDataSymbol: "circle", lineDataSymbolSize: 6, showValue: true, dataLabelFormatCode: "0.00" } },
      { type: pres.charts.LINE, data: [{ name: "損益分岐", labels: years, values: [2.8, 2.8, 2.8, 2.8, 2.8] }], options: { chartColors: [K.gray500], lineSize: 1.5, lineDash: "dash", lineDataSymbol: "none", showValue: false } },
    ], { x: rx, y: top + 0.28, w: rw, h: 1.9, ...axis(0, 5, 1), objectName: "breakeven-chart" });
    txt(s, "損益分岐 2.8（点線）", { x: rx + 0.45, y: top + 0.5, w: 1.6, h: 0.2, size: 8.5, color: K.gray700 });
    txt(s, "713万円（販管費／人）÷255万円（粗利／拠点）＝2.8。4年目に初めて超える", { x: rx, y: top + 2.2, w: rw, h: 0.34, size: S.note, color: K.gray700 });

    box(s, { x: rx, y: top + 2.62, w: rw, h: 1.08, fill: K.ink, name: "funding" });
    txt(s, "必要資金 1.5億円", { x: rx + 0.15, y: top + 2.68, w: rw - 0.3, h: 0.32, size: 13, bold: true, color: K.paper });
    txt(s, [run("谷 0.65億＋運転資金 0.29億＋1年遅延の備え 0.56億", { breakLine: true }), run("調達：電線・ロボットメーカーのCVCから。資金と実証フィールド・販路を同時に得る")], { x: rx + 0.15, y: top + 2.98, w: rw - 0.3, h: 0.68, size: 8.5, color: K.paper, psa: 3 });

    txt(s, "算出根拠：顧客拠点 3→12→30→55→85／売上総利益 5・29・73・135・213（ライセンス粗利率85%、導入支援・受託40%）／販管費 37・56・79・107・144（人件費800万円×社員数 4→20名−原価振替＋経費）", { x: X0, y: 4.92, w: W, h: 0.36, size: S.note, color: K.gray700 });
  }
  // ---------- Slide 8 課題とまとめ ----------
  {
    const s = add(8);
    heading(s, "設問⑤　リスク ─ 最大のリスクは自分。判断基準を先に決める", "創業者が1か月不在でも回る状態を5年目に置き、BSCの4指標で任せる");
    const stages = [
      ["準備期（1年目）", [
        ["製品がない段階での受注", "有償試験導入を3拠点に限定。1拠点に完全対応し横展開の型を先に構築"],
        ["創業者が律速になる", "現場ヒアリングを手順書化。1年目末にCSが単独で運用できる状態へ"],
        ["製造現場の知見不足", "生産技術経験者を初期メンバーに迎え、顧客工場に入って学ぶ"],
      ]],
      ["中期（2〜3年目）", [
        ["個社対応による受託化", "共通機能8割・個社対応2割を明文化。受注を断る基準を保持"],
        ["採用の難しさ", "製造・施工の現場経験者にITを教える経路に限定"],
        ["母数の小ささ", "アンゾフの新市場開拓。施工会社へ同じ基盤を展開、CRMで解約兆候を検知"],
      ]],
      ["後期（4〜5年目）", [
        ["CADDiなど大手基盤の連続工程参入", "図面に現れない現場処置の蓄積と監修人材が障壁"],
        ["顧客の内製化", "囲い込まない方針のもと、基盤の維持と標準化で課金"],
        ["属人経営からの脱却", "BSCでCSF・KPIを明文化し、判断を権限委譲"],
      ]],
    ];
    const cw = 2.98, gap = 0.13, top = 1.12, ch = 2.02;
    stages.forEach(([h, items], i) => {
      const x = X0 + i * (cw + gap);
      box(s, { x, y: top, w: cw, h: ch, line: K.rule, name: "stage-" + i });
      box(s, { x, y: top, w: cw, h: 0.3, fill: K.ink });
      txt(s, h, { x: x + 0.12, y: top, w: cw - 0.24, h: 0.3, size: 10.5, bold: true, color: K.paper, valign: "middle" });
      items.forEach(([t, d], j) => {
        const y = top + 0.36 + j * 0.55;
        txt(s, [run(t, { bold: true, breakLine: true, fontSize: 9.5 }), run("→ " + d, { fontSize: 8.5 })], { x: x + 0.12, y, w: cw - 0.24, h: 0.54 });
      });
    });
    // BSC: the decision criteria that let the founder step back
    const bt = 3.22, bw = (W - 1.0) / 4;
    txt(s, [run("BSC", { bold: true, fontSize: 10.5, breakLine: true }), run("CSF・KPI", { fontSize: 8, color: K.gray700 })], { x: X0, y: bt, w: 1.0, h: 0.56, valign: "middle" });
    [
      ["財務", "1人あたり顧客拠点 2.8以上"],
      ["顧客", "初年度費用の回収 約2.1か月"],
      ["業務プロセス", "導入2か月／共通機能8割"],
      ["学習と成長", "創業者1か月不在で事業継続"],
    ].forEach(([k, v], i) => {
      const x = X0 + 1.0 + i * bw;
      box(s, { x: x + 0.04, y: bt, w: bw - 0.08, h: 0.56, fill: K.fill, name: "bsc-" + i });
      txt(s, [run(k, { bold: true, fontSize: 9, breakLine: true }), run(v, { fontSize: 8.5 })], { x: x + 0.14, y: bt + 0.04, w: bw - 0.28, h: 0.48, valign: "middle" });
    });
    const by = 3.9;
    box(s, { x: X0, y: by, w: W, h: 1.3, fill: K.ink, name: "closing" });
    txt(s, "この事業計画の三つの柱は、すべて現在の勤務先の中にも存在する", { x: X0 + 0.2, y: by + 0.1, w: W - 0.4, h: 0.36, size: 14, bold: true, color: K.paper });
    txt(s, [
      run("顧客＝社内の工場（生産技術・製造部門）　／　商品＝製造記録のデータ化と技能継承　／　収益＝間接工数の削減と品質不良の防止", { breakLine: true }),
      run("復職後は生産技術・情報システムの領域でこの役割を担いたい。役割を定義しきることが、抱え込まないための再発防止策でもある。"),
    ], { x: X0 + 0.2, y: by + 0.52, w: W - 0.4, h: 0.72, size: 10, color: K.paper, psa: 5 });
  }

  fs.mkdirSync(OUT_DIR, { recursive: true });
  await pres.writeFile({ fileName: OUT });
  console.log("wrote", OUT);
}

main().catch((e) => { console.error(e); process.exit(1); });
