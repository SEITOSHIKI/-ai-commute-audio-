// Build the GEOREC business-plan deck (8 slides, speaker notes) with pptxgenjs.
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

// ---------- notes ----------
const NOTES = {
  1: "【0:00 - 0:40】\n株式会社ジオレックの事業構想を発表します。タグラインは「掘る前に、わかる。」です。地中送電線の工事では、掘ってみるまで分からないことが数多くあります。この事業は、過去の工事記録を使って、それを掘る前に分かるようにするものです。右側に、設問①から⑤への結論を一行ずつ示しました。以降、この順に根拠をご説明します。",
  2: "【0:40 - 2:00】\n設問①の一つ目、事業内容です。売るのは記録そのものではなく、記録から導く次の工事の予測です。土台となるのが地中設備データ基盤で、工事、設備、耐圧・絶縁抵抗・部分放電の試験値、埋設位置と深さ、材料ロットを一つのデータモデルにまとめます。竣工と同時に、位置情報つきの設備台帳ができあがります。記録は目的ではなく燃料です。この基盤の上に四つのモジュールを載せます。ROUTEは、試掘の前に埋設物との干渉リスクを区間ごとに示します。WINDOWは、年に数回しか取れない停電作業枠を外す確率を事前に算出します。JOINTは、接続部の施工中に過去の不具合事例と照合し、逸脱をその場で警告します。COSTは、過去の実績原価から想定原価と赤字確率を示します。価格は、基盤100万円、工事登録4万円×年25件、モジュール50万円×平均2本で、1社あたり年300万円です。初年度のみ、導入支援として200万円をいただきます。",
  3: "【2:00 - 3:20】\nこの事業を選んだ理由です。地中送電線は、工事の当日しか姿を見せません。記録の機会は埋め戻すまでの数時間に一度きりで、そこを逃すと二度と取れません。私は古河電気工業で、66kVと275kVの地中送電線の施工管理を5年間担当し、その記録が次の工事に活かされない場面を何度も見てきました。同時に、PythonやPower BI、FastAPIなどを独学し、業務の中で実際に仕組みを作ってきました。現場とITの両方を知っていることが、この事業を選んだ理由です。市場環境は二方向から追い風です。高度成長期の送配電設備が経年対策期を迎え、データセンター需要で設備投資も増えています。電気工事業の許可業者は63,144社、建設テック市場は2030年度に3,000億円を超える見込みです。一方で、ITツールを導入しても42.5%が現場に定着しておらず、54.4%が1日2時間以上を事務作業に費やしています。",
  4: "【3:20 - 4:40】\n最初に自分で自分に突きつけた問いが三つありました。コンサルと何が違うのか、ANDPADと何が違うのか、作図AIが来たらどうなるのか、です。コンサルティングが売るのは人の時間で、課金は人月です。当社が売るのはデータ基盤で、人が引いても台帳と予測モデルが残ります。ANDPADなどの建築向けSaaSは、工事を終わらせるための道具です。当社は置き換えず、併存します。工事中の進捗や写真はANDPAD、竣工後に残す設備データと予測はジオレックです。むしろ導入済みの企業こそ、工事は回るようになったが設備データは残っていない、と気づいているため、見込み顧客になります。私の勤務先でも同じ実感があります。作図AIは描くAIで、その場所に20年前から何が埋まっているかは知りません。描画は任せ、当社はデータを供給する側に回ります。右下のSWOTでは、弱みとして、営業と資金調達が未経験であること、創業者一人への依存を正直に挙げています。",
  5: "【4:40 - 5:40】\n設問③、経営理念です。「掘る前に、わかる。」を理念に置きます。ホームページには次の文を載せます。地中に埋めた送電線は、工事のその日にしか姿を見せません。誰が、どう接続し、どんな試験値だったのか。その記録は20年後、同じ場所を掘り返す技術者にとって、たった一つの手がかりになります。私たちは記録を、過去のためではなく、次の工事のために残します。掘ってみるまで分からなかったことを、掘る前に分かるようにするために。行動指針は三つです。読み手は20年後の技術者であること。現場の手を増やさないこと。そして、データは顧客のものであり、囲い込まないことです。",
  6: "【5:40 - 7:00】\n設問②、組織と採用です。顧客数は希望ではなく、導入担当の人数と生産性から逆算して置いています。1社の導入には、発注者様式の作り込み、過去データの移行、現場研修で約2か月かかります。並行できるのは2社までなので、導入担当1名あたり年6社が上限です。立ち上がりを考慮し、計画値は上限の6割から8割に置きました。1年目は製品が未完成のため、上限9社に対して3社に限定します。これは能力ではなく、意図的な制約です。累計顧客数は5年目に85社、社員は初年度4名から5年目20名とします。採用では、IT人材に現場を教えるより、施工管理経験者にITを教えるほうが早いと考えています。選考では、現場に出ることを厭わないこと、そして作らない判断ができることを見ます。私自身の役割は、自ら現場に出る者から、現場の読み解き方を型にして渡す者へ移していきます。創業者が1か月不在でも事業が回る状態を、5年目の目標とします。",
  7: "【7:00 - 8:30】\n設問④、経営計画です。売上高は1年目1,200万円、5年目に2億9,000万円です。営業利益は3年目まで赤字で、4年目に2,800万円の黒字に転換し、5年目の営業利益率は24%です。なぜ4年目なのか。黒字化の条件は売上規模ではなく、社員1人あたりの顧客数で決まります。4年目の社員1人あたり販管費は713万円、顧客1社あたりの粗利は255万円で、社員1人あたり2.8社が損益分岐です。3年目は2.73社で届かず、4年目は3.67社で超えます。顧客の増加が人員の増加を追い越す年です。必要資金は1.5億円です。3年目末の累積赤字0.65億円、運転資金0.29億円、計画が1年遅れた場合の備え0.56億円を積み上げた結果です。顧客側では、施工管理者100名の会社で年2,880万円の工数削減となり、初年度費用500万円を約2.1か月で回収できます。ただし削減時間が他の業務に振り向けられる前提です。本当の価値は、停電作業枠を外さないことと、波及事故を防ぐことにあります。",
  8: "【8:30 - 9:50】\n設問⑤、ステージごとの経営課題です。最大のリスクは競合ではなく、創業者である自分が律速になることだと考えています。準備期は、製品がない段階での受注と創業者依存が課題です。有償試験導入を3社に限定し、現場ヒアリングを手順書化して、1年目末にはカスタマーサクセスが単独で回せる状態にします。中期は、受託化と採用が課題です。共通機能8割・個社対応2割を明文化し、受注を断る基準を持ちます。後期は、大手の参入と属人経営からの脱却が課題です。最後に、この事業計画の三つの柱である顧客・商品・収益は、すべて現在の勤務先の中にも存在します。顧客は社内の施工管理部門、商品は竣工記録のデータ化と予測、収益は工数削減と調査コストの低減です。だから復職後は、情報システム・生産技術の領域で、現場と技術のあいだを訳す役割を担いたいと考えています。そして、自らの役割を定義しきること自体が、業務を際限なく抱え込まないための再発防止策でもあります。以上です。",
};

async function main() {
  const pres = new pptxgen();
  pres.layout = T.slide.layout;
  pres.title = "株式会社ジオレック 事業構想";
  pres.theme = { headFontFace: F, bodyFontFace: F };

  const markBlack = await logo("georec-mark-black");
  const markWhite = await logo("georec-mark-white");
  const wordWhite = await logo("georec-wordmark-white");

  pres.defineSlideMaster({
    title: "GR_CONTENT",
    background: { color: K.paper },
    objects: [
      { image: { x: 9.42, y: 0.22, w: 0.3, h: 0.29, data: markBlack } },
      { text: { text: "掘る前に、わかる。", options: { x: 6.6, y: 5.3, w: 2.6, h: 0.22, fontFace: F, fontSize: S.footer, color: K.gray500, align: "right", margin: 0, valign: "middle" } } },
    ],
    slideNumber: { x: 9.3, y: 5.3, w: 0.4, h: 0.22, fontFace: F, fontSize: S.footer, color: K.gray500, align: "right", margin: 0 },
  });
  const add = (n) => { const s = pres.addSlide({ masterName: "GR_CONTENT" }); s.addNotes(NOTES[n]); return s; };

  // ---------- Slide 1 表紙 ----------
  {
    const s = add(1);
    box(s, { x: 0, y: 0, w: 3.33, h: 5.625, fill: K.ink, name: "cover-panel" });
    s.addImage({ data: wordWhite, x: 0.35, y: 1.45, w: 2.6, h: 0.64 });
    txt(s, "掘る前に、わかる。", { x: 0.35, y: 2.45, w: 2.9, h: 0.45, size: 20, bold: true, color: K.paper });
    txt(s, "株式会社ジオレック（事業構想）", { x: 0.35, y: 3.0, w: 2.9, h: 0.3, size: 11, color: K.paper });

    const rx = 3.75, rw = 5.55;
    txt(s, "リワーク実習　セルフマネジメント課題", { x: rx, y: 0.5, w: rw, h: 0.28, size: S.lead, color: K.gray700 });
    txt(s, "掘ってみるまで分からなかったことを、掘る前に分かるようにする", { x: rx, y: 0.85, w: rw, h: 0.85, size: 22, bold: true, lsm: 1.2 });
    const sum = [
      ["① 事業", "電力・通信インフラの地中設備を対象とした、予測型データ基盤の提供"],
      ["② 組織", "初年度4名、5年目20名。導入担当1名あたり年6社を設計基準とする"],
      ["③ 理念", "掘る前に、わかる。─ 記録は過去ではなく、次の工事のためにある"],
      ["④ 計画", "4年目に黒字転換。5年目 売上2.9億円・営業利益率24%"],
      ["⑤ 課題", "準備期＝創業者依存　中期＝受託化と採用　後期＝大手参入と属人経営"],
    ];
    const y0 = 2.0, rh = 0.46;
    sum.forEach(([k, v], i) => {
      const y = y0 + i * rh;
      hline(s, rx, y, rw, K.rule, T.stroke.rule);
      txt(s, k, { x: rx, y, w: 0.75, h: rh, size: 10.5, bold: true, valign: "middle" });
      txt(s, v, { x: rx + 0.75, y, w: rw - 0.75, h: rh, size: 10.5, valign: "middle" });
    });
    hline(s, rx, y0 + 5 * rh, rw, K.rule, T.stroke.rule);
    txt(s, "発表者：＿＿＿＿＿＿＿＿＿＿　　発表時間：10分", { x: rx, y: 4.65, w: rw, h: 0.3, size: 10.5, color: K.gray700 });
  }

  // ---------- Slide 2 事業内容 ----------
  {
    const s = add(2);
    heading(s, "設問①-1　事業内容 ─ 記録を燃料に、掘る前の判断を支える", "売るのは記録そのものではない。記録から導く、次の工事の予測である");
    // left: platform
    const lx = X0, lw = 3.45, top = 1.2;
    box(s, { x: lx, y: top, w: lw, h: 2.0, fill: K.fill });
    txt(s, "共通基盤　地中設備データ基盤", { x: lx + 0.15, y: top + 0.12, w: lw - 0.3, h: 0.3, size: 12.5, bold: true });
    txt(s, [
      run("工事・設備・試験値（耐圧／絶縁抵抗／部分放電）・埋設位置／深さ・材料ロットを単一のデータモデルに統合", { breakLine: true }),
      run("竣工と同時に位置情報つき設備台帳が成立", { breakLine: true }),
      run("全モジュールの共通基盤。基盤契約に設備台帳機能を含む"),
    ], { x: lx + 0.15, y: top + 0.48, w: lw - 0.3, h: 1.45, size: 10, psa: 4 });
    box(s, { x: lx, y: top + 2.12, w: lw, h: 0.98, fill: K.ink });
    txt(s, "記録は目的ではなく燃料", { x: lx + 0.15, y: top + 2.2, w: lw - 0.3, h: 0.35, size: 15, bold: true, color: K.paper });
    txt(s, "目的は掘る前の判断。記録が増えるほど予測精度が上がり、使うほど強くなる構造が参入障壁となる", { x: lx + 0.15, y: top + 2.57, w: lw - 0.3, h: 0.48, size: 9.5, color: K.paper });
    txt(s, "提供順：1年目 基盤 → 2年目 ROUTE → 3年目 WINDOW → 4年目 JOINT → 5年目 COST", { x: lx, y: top + 3.15, w: lw, h: 0.38, size: S.note, color: K.gray700 });

    // right: 2x2 modules
    const mods = [
      ["ROUTE", "干渉予測", "試掘時の未知埋設物による設計やり直し・工期延伸", "過去竣工データと道路管理者・占用事業者情報の突合による区間別干渉リスクのスコア化（CADレイヤ出力）"],
      ["WINDOW", "工程リスク予測", "年数回に限られる停電作業枠の逸失（次は1年後）", "実績工数・天候・交通規制・協議リードタイムからの枠内未完了確率の事前算出と遅延要因の特定"],
      ["JOINT", "施工品質アシスト", "接続部の施工不良による波及事故", "施工中の環境条件・手順の記録と、過去不具合事例・試験値トレンドとの照合による逸脱の即時警告"],
      ["COST", "見積支援", "工事条件の都度変化による属人的見積（赤字受注・失注）", "実績原価と工事条件の紐づけによる想定原価レンジと赤字確率の提示"],
    ];
    const gx = 4.05, cw = 2.7, ch = 1.62, gap = 0.15;
    mods.forEach(([name, jp, prob, offer], i) => {
      const x = gx + (i % 2) * (cw + gap), y = top + Math.floor(i / 2) * (ch + gap);
      box(s, { x, y, w: cw, h: ch, line: K.rule, name: "module-" + name });
      txt(s, [run(name, { bold: true, fontSize: 13 }), run("　" + jp, { fontSize: 9.5, color: K.gray700 })], { x: x + 0.12, y: y + 0.08, w: cw - 0.24, h: 0.3, valign: "middle" });
      txt(s, [run("課題　", { bold: true }), run(prob)], { x: x + 0.12, y: y + 0.44, w: cw - 0.24, h: 0.38, size: 9.5 });
      txt(s, [run("提供　", { bold: true }), run(offer)], { x: x + 0.12, y: y + 0.86, w: cw - 0.24, h: 0.7, size: 9.5 });
    });
    // price strip
    box(s, { x: X0, y: 4.8, w: W, h: 0.38, fill: K.fill });
    hline(s, X0, 4.8, W, K.ink, T.stroke.strong);
    txt(s, [
      run("基盤100万円＋工事登録4万円×25件＋モジュール50万円×平均2本 ＝ "),
      run("1社あたり年300万円", { bold: true }),
      run("（初年度のみ導入支援200万円）"),
    ], { x: X0 + 0.15, y: 4.8, w: W - 0.3, h: 0.38, size: 10.5, valign: "middle" });
  }

  // ---------- Slide 3 選んだ理由 ----------
  {
    const s = add(3);
    heading(s, "設問①-2　この事業を選んだ理由 ─ 原体験と市場環境", "記録の機会は、埋め戻すまでの数時間に一度きり。そこを逃すと二度と取れない");
    const lx = X0, lw = 4.75, top = 1.2;
    txt(s, "解くべき課題", { x: lx, y: top, w: lw, h: 0.28, size: 12, bold: true });
    txt(s, "地中送電線は工事の当日のみ露出。記録の機会は埋め戻しまでの数時間に一度きりで、逸すると再取得は不可能。蓄積した記録を次の工事の予測に転用し、記録の増加が予測精度を高める構造を参入障壁とする", { x: lx, y: top + 0.3, w: lw, h: 0.72, size: 10 });
    txt(s, "原体験", { x: lx, y: top + 1.08, w: lw, h: 0.28, size: 12, bold: true });
    txt(s, [
      run("66kV／275kV 地中送電線の施工管理 5年", { breakLine: true }),
      run("Python・Power BI・FastAPI・Docker・LangChain・RAG の独学と業務実装", { breakLine: true }),
      run("1級電気工事施工管理技士・G検定"),
    ], { x: lx, y: top + 1.38, w: lw, h: 1.0, size: 10, psa: 2 });
    hline(s, lx, top + 2.45, lw, K.rule, T.stroke.rule);
    txt(s, "追い風は二方向から", { x: lx, y: top + 2.55, w: lw, h: 0.28, size: 12, bold: true });
    [["更新", "高度成長期の送配電設備が本格的な経年対策期へ"], ["新設", "データセンター需要による送配電設備投資の増加局面"]].forEach(([k, v], i) => {
      const y = top + 2.86 + i * 0.36;
      box(s, { x: lx, y: y + 0.04, w: 0.5, h: 0.26, fill: K.ink });
      txt(s, k, { x: lx, y: y + 0.04, w: 0.5, h: 0.26, size: 9.5, bold: true, color: K.paper, align: "center", valign: "middle" });
      txt(s, v, { x: lx + 0.62, y, w: lw - 0.62, h: 0.34, size: 10.5, valign: "middle" });
    });
    // stats
    const stats = [
      ["63,144社", "電気工事業の建設業許可業者数"],
      ["3,000億円超", "建設テック市場規模（2023年度実績 1,845億円 → 2030年度予測）"],
      ["42.5%", "ITツール導入済みでも現場への定着が不十分"],
      ["54.4%", "1日2時間以上を事務作業に費やす建設業従事者"],
    ];
    const gx = 5.4, cw = 2.05, ch = 1.62, gap = 0.1;
    stats.forEach(([fig, lab], i) => {
      const x = gx + (i % 2) * (cw + gap), y = top + Math.floor(i / 2) * (ch + gap);
      box(s, { x, y, w: cw, h: ch, line: K.rule, name: "stat-" + i });
      txt(s, fig, { x: x + 0.12, y: y + 0.15, w: cw - 0.24, h: 0.5, size: 20, bold: true, valign: "middle" });
      txt(s, lab, { x: x + 0.12, y: y + 0.75, w: cw - 0.24, h: 0.8, size: 9.5 });
    });
    txt(s, "出典：国土交通省 建設業許可業者数調査（令和5年3月末）／矢野経済研究所／サイボウズ 建設業従事者調査（2026年、n=1,000）／電力広域的運営推進機関／経済産業省 局地的電力需要増加と送配電ネットワークに関する研究会（2024年6月）", { x: X0, y: 4.85, w: W, h: 0.38, size: S.note, color: K.gray700 });
  }

  // ---------- Slide 4 競合分析 ----------
  {
    const s = add(4);
    heading(s, "競合分析 ─ コンサルでも、建築向けSaaSでも、作図AIでもない", "既存サービスを置き換えない。工事中は既存ツール、掘る前の判断は当社が担う");
    const rows = [
      ["", "コンサルティング", "建築向けSaaS", "CAD生成AI", "ジオレック"],
      ["代表例", "アクセンチュア 等", "ANDPAD／SPIDERPLUS", "テキスト→図面化AI", "─"],
      ["できること", "人の時間と提言", "工事中の情報共有", "図面を新しく描く", { text: "過去の記録から次を予測", bold: true }],
      ["持っていないもの", "業界実務の経験", "試験値・設備データの器", "その場所の20年分の履歴", "─"],
      ["課金形態", "人月（フロー）", "ID課金", "従量／ライセンス", { text: "基盤＋工事件数＋モジュール", bold: true }],
      ["対象領域", "全業種", "建築・住宅", "作図全般", { text: "電力・通信の地中設備", bold: true }],
    ];
    table(s, rows, { x: X0, y: 1.15, w: W, colW: [1.4, 1.85, 1.95, 1.85, 2.15], rowH: 0.29, size: 9.5, align: "left", emCol: 4, name: "competitors" });

    const by = 3.1;
    box(s, { x: X0, y: by, w: 4.45, h: 1.75, line: K.rule });
    txt(s, "既存SaaS導入済みの企業こそ、見込み顧客になる", { x: X0 + 0.15, y: by + 0.1, w: 4.15, h: 0.3, size: 11.5, bold: true });
    txt(s, [
      run("置き換えず併存：", { bold: true }), run("工事中の進捗・写真は既存SaaS、竣工後に残す設備データと予測は当社", { breakLine: true }),
      run("導入済み企業の認識：", { bold: true }), run("「工事は回るようになったが、設備データは残っていない」", { breakLine: true }),
      run("CAD生成AIは補完：", { bold: true }), run("描画はAIに任せ、その場所の履歴データを当社が供給"),
    ], { x: X0 + 0.15, y: by + 0.45, w: 4.15, h: 1.25, size: 9.5, psa: 4 });

    const sx = 5.05, sw = 4.55;
    txt(s, "SWOT", { x: sx, y: by, w: sw, h: 0.26, size: 11.5, bold: true });
    [
      ["S", "施工管理5年と自作開発の両立／発注者様式・試験記録の実務知識／競合が嫌う領域を苦にしない"],
      ["W", "営業・資金調達・組織運営が未経験／実績と知名度がゼロ／創業者一人への依存"],
      ["O", "設備の経年対策期と新設需要の重複／生成AIによる非構造データ活用コストの低下"],
      ["T", "大手の領域参入／顧客の内製化／母数が小さく1社の解約が重い／採用競争"],
    ].forEach(([k, v], i) => {
      const y = by + 0.32 + i * 0.36;
      box(s, { x: sx, y: y + 0.03, w: 0.28, h: 0.28, fill: K.ink });
      txt(s, k, { x: sx, y: y + 0.03, w: 0.28, h: 0.28, size: 9.5, bold: true, color: K.paper, align: "center", valign: "middle" });
      txt(s, v, { x: sx + 0.38, y, w: sw - 0.38, h: 0.34, size: 8.5, valign: "middle" });
    });
    txt(s, "※ SPIDERPLUS の実績KPI：契約1,593社／ARPU 3,971円・ID月／解約率0.6%（SpiderPlus & Co. 決算説明資料）", { x: X0, y: 4.95, w: W, h: 0.25, size: S.note, color: K.gray700 });
  }

  // ---------- Slide 5 経営理念 ----------
  {
    const s = add(5);
    heading(s, "設問③　経営理念", null);
    box(s, { x: X0, y: 0.85, w: W, h: 1.15, fill: K.ink, name: "philosophy-band" });
    s.addImage({ data: markWhite, x: 0.8, y: 1.0, w: 0.88, h: 0.85 });
    txt(s, "掘る前に、わかる。", { x: 2.0, y: 0.85, w: 7.3, h: 1.15, size: S.taglineDisplay, bold: true, color: K.paper, valign: "middle" });

    txt(s, "ホームページ掲載文", { x: X0, y: 2.15, w: W, h: 0.25, size: 9.5, bold: true, color: K.gray700 });
    txt(s, [
      run("地中に埋めた送電線は、工事のその日にしか姿を見せません。", { breakLine: true }),
      run("誰が、どう接続し、どんな試験値だったのか。その記録は20年後、同じ場所を掘り返す技術者にとって、たった一つの手がかりになります。", { breakLine: true }),
      run("私たちは記録を、過去のためではなく、次の工事のために残します。掘ってみるまで分からなかったことを、掘る前に分かるようにするために。"),
    ], { x: X0, y: 2.42, w: W, h: 1.15, size: 11.5, psa: 2 });

    const cards = [
      ["1", "読み手は20年後の技術者", "記録は今の担当者のためではなく、将来その設備を掘り返す人のために書く。判断の根拠まで残す"],
      ["2", "現場の手を増やさない", "記録のために作業を足した時点で失敗とみなす。すでに行っている作業の延長からデータを取る"],
      ["3", "データは顧客のものである", "囲い込まない。契約終了時は全データを標準形式で返却。残る理由は拘束ではなく有用性に置く"],
    ];
    const cw = 2.97, gap = 0.145, cy = 3.72, ch = 1.42;
    cards.forEach(([n, t, d], i) => {
      const x = X0 + i * (cw + gap);
      box(s, { x, y: cy, w: cw, h: ch, fill: K.fill, name: "principle-" + n });
      txt(s, [run("行動指針 " + n + "　", { fontSize: 9.5, color: K.gray700 })], { x: x + 0.15, y: cy + 0.1, w: cw - 0.3, h: 0.22 });
      txt(s, t, { x: x + 0.15, y: cy + 0.32, w: cw - 0.3, h: 0.32, size: 13, bold: true });
      txt(s, d, { x: x + 0.15, y: cy + 0.7, w: cw - 0.3, h: 0.66, size: 9.5 });
    });
  }

  // ---------- Slide 6 組織と採用 ----------
  {
    const s = add(6);
    heading(s, "設問②　組織と採用 ─ 顧客獲得能力を人員から設計する", "顧客数は希望ではなく、導入担当の人数と生産性から逆算して置いている");
    const lx = X0, lw = 4.5, top = 1.15;
    box(s, { x: lx, y: top, w: lw, h: 1.0, line: K.ink, lw: T.stroke.strong });
    txt(s, "設計基準　導入担当1名あたり 年6社", { x: lx + 0.15, y: top + 0.08, w: lw - 0.3, h: 0.32, size: 13, bold: true });
    txt(s, "1社の導入＝発注者様式の作り込み・過去データ移行・現場研修で約2か月。並行2社が上限 → 12か月÷2か月×1社＝年6社。計画値は上限の6〜8割", { x: lx + 0.15, y: top + 0.42, w: lw - 0.3, h: 0.54, size: 9.5 });
    table(s, [
      ["", "1年目", "2年目", "3年目", "4年目", "5年目"],
      ["導入担当（名）", "1.5", "2.5", "3.5", "5.5", "7.0"],
      ["理論上限（×6社）", "9", "15", "21", "33", "42"],
      ["新規獲得（計画）", "3", "9", "18", "27", "33"],
      ["解約", "0", "0", "0", "2", "3"],
      ["累計顧客数", "3", "12", "30", "55", "85"],
    ], { x: lx, y: top + 1.15, w: lw, colW: [1.5, 0.6, 0.6, 0.6, 0.6, 0.6], rowH: 0.27, strongRows: [5], name: "customers" });
    txt(s, "※1年目は製品が未完成のため、上限9社に対し3社に限定する（能力ではなく意図的な制約）", { x: lx, y: top + 2.85, w: lw, h: 0.3, size: S.note, color: K.gray700 });

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
    txt(s, "採用基準と代表の役割", { x: rx, y: top + 2.08, w: rw, h: 0.28, size: 12, bold: true });
    txt(s, [
      run("採用の軸：", { bold: true }), run("施工管理経験者にITを教える経路（逆経路より短期間）", { breakLine: true }),
      run("選考基準：", { bold: true }), run("現場に出ることを厭わない／「作らない」判断ができる", { breakLine: true }),
      run("代表の役割：", { bold: true }), run("自ら現場に出る者 → 現場の読み解き方を型にして渡す者", { breakLine: true }),
      run("5年目の到達目標：", { bold: true }), run("創業者が1か月不在でも事業が回る状態"),
    ], { x: rx, y: top + 2.4, w: rw, h: 1.6, size: 10, psa: 4 });
  }

  // ---------- Slide 7 経営計画 ----------
  {
    const s = add(7);
    heading(s, "設問④　経営計画 ─ 5年収支、黒字化の条件、必要資金", "黒字化の条件は売上規模ではなく、社員1人あたりの顧客数で決まる");
    table(s, [
      ["単位：百万円", "1年目", "2年目", "3年目", "4年目", "5年目", "算出根拠"],
      ["累計顧客数（社）", "3", "12", "30", "55", "85", "新規獲得−解約の累計（導入担当×年6社の6〜8割）"],
      ["売上高", "12", "48", "110", "190", "290", "ライセンス＋導入支援200万円／社＋受託開発"],
      ["売上総利益", "5", "29", "73", "135", "213", "粗利率 ライセンス85%・導入支援／受託40%"],
      ["販管費", "37", "56", "79", "107", "144", "人件費800万円×社員数−原価振替＋その他経費"],
      ["営業利益", "▲32", "▲27", "▲6", "28", "69", "売上総利益−販管費"],
      ["営業利益率", "─", "─", "─", "15%", "24%", "営業利益÷売上高"],
      ["社員数（名）", "4", "7", "11", "15", "20", "代表＋エンジニア＋CS＋営業＋管理"],
    ], { x: X0, y: 1.12, w: W, colW: [1.45, 0.66, 0.66, 0.66, 0.66, 0.66, 4.45], rowH: 0.245, strongRows: [5], leftCols: [6], mutedCol: 6, name: "pl" });

    const boxes = [
      ["なぜ4年目に黒字化するか", [
        "販管費／人　107百万円÷15名＝713万円",
        "粗利／社　300万円×85%＝255万円",
        "損益分岐　713÷255＝2.8社／人",
        "3年目　30社÷11名＝2.73（未達）",
        "4年目　55社÷15名＝3.67（超過）",
      ], "顧客の増加が人員の増加を追い越す年"],
      ["必要資金 1.5億円の内訳", [
        "累積赤字ピーク（3年目末）　0.65億円",
        "運転資金（116百万円×3か月）　0.29億円",
        "1年遅延への備え　0.56億円",
        "合計　1.50億円",
      ], "5年間の累積営業損益は＋32百万円。必要額の積み上げ結果"],
      ["顧客側の投資回収（施工管理者100名）", [
        "時間単価　780万円÷1,950h＝4,000円",
        "年間削減　6h×12か月×100名×4,000円",
        "　＝2,880万円",
        "初年度費用　500万円",
        "回収期間 約2.1か月／効果倍率 5.8倍",
      ], "削減時間の再配分が前提。本質的価値は停電枠の確保と波及事故の防止"],
    ];
    const cw = 2.98, gap = 0.13, by = 3.2, bh = 1.68;
    boxes.forEach(([t, lines, note], i) => {
      const x = X0 + i * (cw + gap);
      box(s, { x, y: by, w: cw, h: bh, line: K.rule, name: "calc-" + i });
      txt(s, t, { x: x + 0.12, y: by + 0.08, w: cw - 0.24, h: 0.28, size: 10.5, bold: true });
      txt(s, lines.map((l, j) => run(l, { breakLine: j < lines.length - 1, bold: i === 1 && j === 3 })), { x: x + 0.12, y: by + 0.4, w: cw - 0.24, h: 1.24, size: 8.5, psa: 2 });
      txt(s, note, { x, y: by + bh + 0.04, w: cw, h: 0.32, size: S.note, color: K.gray700 });
    });
  }

  // ---------- Slide 8 課題とまとめ ----------
  {
    const s = add(8);
    heading(s, "設問⑤　ステージごとに想定される経営課題", "最大のリスクは競合ではなく、創業者である自分が律速になること");
    const stages = [
      ["準備期（1年目）", [
        ["製品がない段階での受注", "有償試験導入を3社に限定。1社の様式に完全対応し、横展開の型を先に構築"],
        ["創業者が律速になる", "現場ヒアリングを手順書化。1年目末にCSが単独で運用できる状態へ"],
        ["発注者側の様式変更", "様式定義をデータ構造から分離し、設定変更で吸収"],
      ]],
      ["中期（2〜3年目）", [
        ["個社対応による受託化", "共通機能8割・個社対応2割を明文化。受注を断る基準を保持"],
        ["採用の難しさ", "施工管理経験者にITを教える経路に限定"],
        ["母数の小ささ", "対象3,000社。利用ログによる解約兆候の早期検知"],
      ]],
      ["後期（4〜5年目）", [
        ["大手の領域参入", "蓄積した設備データが障壁。機能は模倣可能、記録の履歴は移設不能"],
        ["顧客の内製化", "囲い込まない方針のもと、基盤の維持と標準化で課金"],
        ["属人経営からの脱却", "権限委譲と意思決定基準の明文化"],
      ]],
    ];
    const cw = 2.98, gap = 0.13, top = 1.12, ch = 2.45;
    stages.forEach(([h, items], i) => {
      const x = X0 + i * (cw + gap);
      box(s, { x, y: top, w: cw, h: ch, line: K.rule, name: "stage-" + i });
      box(s, { x, y: top, w: cw, h: 0.3, fill: K.ink });
      txt(s, h, { x: x + 0.12, y: top, w: cw - 0.24, h: 0.3, size: 10.5, bold: true, color: K.paper, valign: "middle" });
      items.forEach(([t, d], j) => {
        const y = top + 0.4 + j * 0.68;
        txt(s, [run(t, { bold: true, breakLine: true, fontSize: 10 }), run("→ " + d, { fontSize: 9 })], { x: x + 0.12, y, w: cw - 0.24, h: 0.64 });
      });
    });
    const by = 3.72;
    box(s, { x: X0, y: by, w: W, h: 1.45, fill: K.ink, name: "closing" });
    txt(s, "この事業計画の三つの柱は、すべて現在の勤務先の中にも存在する", { x: X0 + 0.2, y: by + 0.12, w: W - 0.4, h: 0.36, size: 14, bold: true, color: K.paper });
    txt(s, [
      run("顧客＝社内の施工管理部門　／　商品＝竣工記録のデータ化と予測　／　収益＝年約2,880万円相当の工数削減と更新工事の調査コスト低減", { breakLine: true }),
      run("復職後は情報システム・生産技術の領域でこの役割を担いたい。役割を定義しきることが、抱え込まないための再発防止策でもある。"),
    ], { x: X0 + 0.2, y: by + 0.58, w: W - 0.4, h: 0.8, size: 10, color: K.paper, psa: 6 });
  }

  fs.mkdirSync(OUT_DIR, { recursive: true });
  await pres.writeFile({ fileName: OUT });
  console.log("wrote", OUT);
}

main().catch((e) => { console.error(e); process.exit(1); });
