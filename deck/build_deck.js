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

// ---------- notes ----------
const NOTES = {
  1: "【0:00 - 0:40】\n株式会社三現ワークスの事業構想を発表します。三現とは、現場・現物・現実のことです。工場でも施工現場でも担い手の高齢化と減少が進み、熟練者の勘が記録に残らないまま失われつつあります。この会社は、その一次情報をデータとAIで次の担い手と機械に引き継ぎ、人が減っても現場が迷わず回る仕組みをつくります。右側に設問①から⑤への結論を示しました。まず、この会社が何のために存在するのかからご説明します。",
  2: "【0:40 - 1:40】\n設問③、経営理念です。Purpose、つまり存在意義は「現場の勘を、次の担い手へ。」です。工場にも施工現場にも、この音がしたら止める、この湿度なら条件を少し変える、といった記録に残らない判断があります。その勘は、担い手が減るいま、退職とともに静かに消えていきます。Missionは、現場・現物・現実の一次情報をデータとAIで引き継ぎ、人が減っても迷わず回る現場をつくることです。Visionとして、5年後に85拠点で熟練者の判断が次の担い手と機械に引き継がれている状態を目指します。行動指針は社名の三現そのもので、現場に立つ、現物に触れる、現実で判断する、の三つです。",
  3: "【1:40 - 3:00】\nこの事業を選んだ理由です。工場でも施工現場でも、条件出しや段取り、異常の見極めは熟練者の勘に頼っています。数字で見ると、建設業の就業者は55歳以上が36.7%、29歳以下は11.7%です。製造業の就業者は2025年に1,033万人まで減り、技能継承がうまくいっていない企業は3社に2社にのぼります。一方、技能継承にデジタル技術を使っている企業は21.7%にとどまります。私は古河電気工業で、地中送電線の施工管理を5年間担当し、ケーブルが使われる現場で不具合や試験値を見てきました。同時に、PythonやFastAPI、RAGなどを独学し、業務の中で仕組みを作ってきました。製品が使われる現場を知り、自分で仕組みを作れることが、この事業を選んだ理由です。",
  4: "【3:00 - 4:20】\n設問①、事業内容です。売るのは記録そのものではなく、熟練者の判断を次の人と機械に渡す仕組みです。土台は、製造ロット、工程条件、試験値、作業記録、熟練者の判断理由を一つにまとめる現場データ基盤です。製品戦略は、基盤を先に入れ、モジュールを年に一本ずつ足していくランド・アンド・エクスパンドです。中心となるKNACKは、ナレッジマネジメントでいうSECIモデルの表出化、つまり熟練者の暗黙知を形式知にする部分を仕組みにしたものです。価格戦略は、原価ではなく顧客の得る価値から決める価値基準価格です。右のグラフのとおり、対象者100名の拠点で毎月240万円の効果が積み上がり、約2.1か月で初年度費用500万円を上回ります。年300万円は、年間効果2,880万円の約1割にあたります。ただし月6時間の削減は、建設業の書類電子化の実績を準用した仮定です。",
  5: "【4:20 - 5:40】\n最初に自分で自分に突きつけた問いが三つありました。コンサルと何が違うのか、MESや大手SIerと何が違うのか、ロボットやAIが来たら人の判断は要らなくなるのではないか、です。左のポジショニングマップは、横軸が扱う情報、つまり何が起きたかという実績か、なぜそうしたかという判断の理由か。縦軸が提供の形、人の時間か仕組みか、です。MESやロボットは実績と動作を仕組みで扱い、コンサルや社内の熟練者は判断の理由を人で扱っています。判断の理由を仕組みで扱う右上が空いており、ここを取ります。STPでは、製造業を工程の形で分け、熟練者の判断が品質を左右する電線・ケーブル工場を狙います。競争戦略は、コトラーの競争地位でいうニッチャー、ポーターの基本戦略でいう差別化集中です。ERPやMESは置き換えずに連携します。弱みとして、製造現場の経験が浅いことと、営業・資金調達が未経験であることを正直に挙げています。",
  6: "【5:40 - 6:50】\n設問②、組織と採用です。顧客数は希望ではなく、導入担当の人数から逆算しています。1拠点の導入には約2か月かかり、並行は2拠点までなので、導入担当1名あたり年6拠点が上限です。計画値はその6割から8割に置き、1年目は製品が未完成のため3拠点に限定します。累計の顧客拠点は5年目に85、社員は4名から20名とします。採用では、製造や施工の現場経験者にITを教える経路を軸にし、製造知見の不足を補うため、生産技術の経験者を初期メンバーに迎えます。私は代表取締役CEOとして、現場の読み解き方を型にして渡す役割に絞ります。技術の責任者は兼任せず、エンジニアの中から任命します。",
  7: "【6:50 - 8:30】\n設問④、経営計画です。グラフの横軸が年、縦軸が金額です。売上高は1年目1,200万円から、5年目に2億9,000万円まで伸びます。営業利益は3年目まで赤字で、4年目に2,800万円の黒字に転じます。点線の累積営業損益は3年目に6,500万円の赤字で底を打ちます。この谷を越えるための資金が1.5億円で、谷の0.65億円、運転資金0.29億円、1年遅れた場合の備え0.56億円の積み上げです。調達先には、電線メーカーやロボットメーカーのコーポレートベンチャーキャピタルを想定しています。資金だけでなく、実証の現場と販路を同時に得られるためです。なぜ4年目に黒字化するのか。右のグラフのとおり、条件は社員1人あたりの顧客拠点数で、損益分岐は2.8です。3年目は2.73で届かず、4年目に3.67で超えます。顧客の増加が人員の増加を追い越す年です。",
  8: "【8:30 - 9:50】\n設問⑤、経営課題です。最大のリスクは、創業者である自分が律速になることです。準備期は創業者依存と製造知見の不足に、試験導入の限定と生産技術経験者の採用で対応します。中期の母数の小ささには、アンゾフの成長マトリクスでいう新市場開拓として、施工会社へ同じ基盤を広げます。後期は属人経営からの脱却です。そのための仕組みが、下段のバランスト・スコアカードです。財務、顧客、業務プロセス、学習と成長の四つの視点で、重要成功要因と指標を先に決めておけば、私がいなくても判断できます。最後に、この三つの柱、顧客・商品・収益は、すべて現在の勤務先の中にも存在します。だから復職後は、生産技術と情報システムの領域で、現場と技術のあいだを訳す役割を担いたいと考えています。そして、自らの役割を定義しきること自体が、業務を際限なく抱え込まないための再発防止策でもあります。以上です。",
};

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

  // ---------- Slide 1 表紙 ----------
  {
    const s = add(1);
    box(s, { x: 0, y: 0, w: 3.33, h: 5.625, fill: K.ink, name: "cover-panel" });
    s.addImage({ data: wordWhite, x: 0.3, y: 1.4, w: 2.75, h: 0.66 });
    txt(s, "現場の勘を、\n次の担い手へ。", { x: 0.35, y: 2.35, w: 2.9, h: 0.8, size: 20, bold: true, color: K.paper, lsm: 1.2 });
    txt(s, "株式会社三現ワークス（事業構想）", { x: 0.35, y: 3.3, w: 2.9, h: 0.3, size: 11, color: K.paper });
    txt(s, "三現＝現場・現物・現実", { x: 0.35, y: 3.62, w: 2.9, h: 0.28, size: 9.5, color: K.paper });

    const rx = 3.75, rw = 5.55;
    txt(s, "リワーク実習　セルフマネジメント課題", { x: rx, y: 0.5, w: rw, h: 0.28, size: S.lead, color: K.gray700 });
    txt(s, "人が減っても、\n現場が迷わず回る仕組みをつくる", { x: rx, y: 0.85, w: rw, h: 0.85, size: 22, bold: true, lsm: 1.2 });
    const sum = [
      ["① 事業", "工場と施工現場の一次情報を、技能継承と自動化につなぐデータ基盤"],
      ["② 組織", "初年度4名、5年目20名。導入担当1名あたり年6拠点が設計基準"],
      ["③ 理念", "現場の勘を、次の担い手へ ─ 現場・現物・現実から始める"],
      ["④ 計画", "4年目に黒字転換。5年目 売上2.9億円・営業利益率24%"],
      ["⑤ 課題", "準備期＝創業者依存と製造知見　中期＝受託化と採用　後期＝大手参入"],
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

  // ---------- Slide 2 Purpose（設問③） ----------
  {
    const s = add(2);
    heading(s, "設問③　経営理念 ─ 現場の勘を、次の担い手へ", "存在意義・使命・5年後の姿・行動指針を一本の言葉から組み立てる");
    box(s, { x: X0, y: 1.12, w: W, h: 1.0, fill: K.ink, name: "purpose-band" });
    s.addImage({ data: markWhite, x: 0.7, y: 1.24, w: 0.76, h: 0.76 });
    txt(s, "PURPOSE　存在意義", { x: 1.75, y: 1.2, w: 7.6, h: 0.24, size: 9.5, bold: true, color: K.paper });
    txt(s, TAGLINE, { x: 1.75, y: 1.42, w: 7.6, h: 0.62, size: 30, bold: true, color: K.paper, valign: "middle" });

    const rows = [
      ["MISSION", "使命", "現場・現物・現実の一次情報をデータとAIで引き継ぎ、人が減っても迷わず回る現場をつくる"],
      ["VISION", "5年後の姿", "85拠点の現場で、熟練者の判断が次の担い手と機械に引き継がれている。売上2.9億円・営業利益率24%"],
      ["PHILOSOPHY", "創業の想い", "記録は目的ではなく燃料。熟練者の勘を、退職とともに消えるものから、次の人が使える資産に変える"],
    ];
    rows.forEach(([en, jp, v], i) => {
      const y = 2.25 + i * 0.47;
      hline(s, X0, y, W, K.rule, T.stroke.rule);
      txt(s, [run(en, { bold: true, fontSize: 10.5, breakLine: true }), run(jp, { fontSize: 8.5, color: K.gray700 })], { x: X0, y: y + 0.04, w: 1.5, h: 0.42, valign: "middle" });
      txt(s, v, { x: X0 + 1.6, y: y + 0.04, w: W - 1.6, h: 0.42, size: 10.5, valign: "middle" });
    });
    hline(s, X0, 2.25 + 3 * 0.47, W, K.rule, T.stroke.rule);

    txt(s, "VALUES　行動指針（三現）", { x: X0, y: 3.78, w: W, h: 0.24, size: 9.5, bold: true, color: K.gray700 });
    const cards = [
      ["現場に立つ", "仕組みを作る前に、その現場で作業を見る。記録のために現場の手を増やさない"],
      ["現物に触れる", "帳票ではなく実物と実測値から始める。熟練者がなぜそう判断したかまで残す"],
      ["現実で判断する", "期待ではなく実績で計画を組み替える。データは顧客のもの。囲い込まない"],
    ];
    const cw = 2.97, gap = 0.145, cy = 4.05, ch = 1.12;
    cards.forEach(([t, d], i) => {
      const x = X0 + i * (cw + gap);
      box(s, { x, y: cy, w: cw, h: ch, fill: K.fill, name: "value-" + (i + 1) });
      txt(s, t, { x: x + 0.15, y: cy + 0.08, w: cw - 0.3, h: 0.32, size: 13, bold: true });
      txt(s, d, { x: x + 0.15, y: cy + 0.44, w: cw - 0.3, h: 0.62, size: 9.5 });
    });
  }

  // ---------- Slide 3 選んだ理由 ----------
  {
    const s = add(3);
    heading(s, "設問①-2　選んだ理由 ─ 熟練者の勘は、記録に残らず消えていく", "担い手の高齢化と減少に対し、技能継承へのデジタル活用は2割にとどまる");
    const lx = X0, lw = 4.75, top = 1.2;
    txt(s, "解くべき課題", { x: lx, y: top, w: lw, h: 0.28, size: 12, bold: true });
    txt(s, "工場でも施工現場でも、条件出し・段取り・異常の見極めは熟練者の勘に依存。担い手の高齢化と減少により、その一次情報が記録に残らないまま失われつつある。人が減る前提で判断を記録とデータで支え、機械に任せる範囲を広げる必要がある", { x: lx, y: top + 0.3, w: lw, h: 0.95, size: 10 });
    txt(s, "原体験", { x: lx, y: top + 1.3, w: lw, h: 0.28, size: 12, bold: true });
    txt(s, [
      run("66kV／275kV 地中送電線の施工管理 5年：製品が使われる現場の不具合と試験値を実地で把握", { breakLine: true }),
      run("Python・Power BI・FastAPI・Docker・LangChain・RAG の独学と業務実装", { breakLine: true }),
      run("1級電気工事施工管理技士・G検定"),
    ], { x: lx, y: top + 1.6, w: lw, h: 1.0, size: 10, psa: 2 });
    hline(s, lx, top + 2.65, lw, K.rule, T.stroke.rule);
    txt(s, "追い風は二方向から", { x: lx, y: top + 2.72, w: lw, h: 0.28, size: 12, bold: true });
    [["技術", "フィジカルAI・自動搬送ロボット・デジタルツインの実用化"], ["需要", "設備の経年対策・データセンター向けの電線需要増"]].forEach(([k, v], i) => {
      const y = top + 3.02 + i * 0.32;
      box(s, { x: lx, y: y + 0.03, w: 0.5, h: 0.24, fill: K.ink });
      txt(s, k, { x: lx, y: y + 0.03, w: 0.5, h: 0.24, size: 9.5, bold: true, color: K.paper, align: "center", valign: "middle" });
      txt(s, v, { x: lx + 0.62, y, w: lw - 0.62, h: 0.3, size: 10, valign: "middle" });
    });
    const stats = [
      ["36.7%", "建設業就業者のうち55歳以上（29歳以下は11.7%）"],
      ["1,033万人", "製造業就業者数（2025年。2023年の1,055万人から減少）"],
      ["3社に2社", "技能継承がうまくいっていない製造企業"],
      ["21.7%", "技能継承にデジタル技術を活用している企業"],
    ];
    const gx = 5.4, cw = 2.05, ch = 1.62, gap = 0.1;
    stats.forEach(([fig, lab], i) => {
      const x = gx + (i % 2) * (cw + gap), y = top + Math.floor(i / 2) * (ch + gap);
      box(s, { x, y, w: cw, h: ch, line: K.rule, name: "stat-" + i });
      txt(s, fig, { x: x + 0.12, y: y + 0.15, w: cw - 0.24, h: 0.5, size: 20, bold: true, valign: "middle" });
      txt(s, lab, { x: x + 0.12, y: y + 0.75, w: cw - 0.24, h: 0.8, size: 9.5 });
    });
    txt(s, "出典：国土交通省（総務省「労働力調査」2024年）／経済産業省・厚生労働省・文部科学省「ものづくり白書」2026年版／労働政策研究・研修機構「ものづくり産業における人材確保・定着と技能継承に関する調査」（2026年5月）", { x: X0, y: 4.9, w: W, h: 0.38, size: S.note, color: K.gray700 });
  }

  // ---------- Slide 4 事業内容（設問①-1） ----------
  {
    const s = add(4);
    heading(s, "設問①-1　事業内容 ─ 熟練者の判断を、次の人と機械に渡す", "1拠点あたり年300万円。顧客は約2.1か月で初年度費用を回収できる");
    const lx = X0, lw = 4.45, top = 1.15;
    box(s, { x: lx, y: top, w: lw, h: 0.72, fill: K.ink });
    txt(s, "共通基盤　現場データ基盤", { x: lx + 0.15, y: top + 0.06, w: lw - 0.3, h: 0.28, size: 12, bold: true, color: K.paper });
    txt(s, "製造ロット・工程条件・試験値・作業記録・熟練者の判断理由を単一のデータモデルに統合", { x: lx + 0.15, y: top + 0.34, w: lw - 0.3, h: 0.34, size: 9.5, color: K.paper });
    txt(s, [run("製品戦略：", { bold: true }), run("基盤が先、モジュールは年1本ずつ追加（ランド＆エクスパンド）")], { x: lx, y: top + 0.78, w: lw, h: 0.24, size: 8.5, color: K.gray700, valign: "middle" });
    const mods = [
      ["TRACE", "ロット追跡", "2年目", "材料ロットから出荷試験、納入先の施工記録まで一本で追跡"],
      ["KNACK", "技能継承（KM）", "3年目", "SECIモデルの表出化。熟練者の判断理由を形式知にし、推奨値を提示"],
      ["TWIN", "工程デジタルツイン", "4年目", "欠員・設備停止時の計画組み替えの影響を事前に試算"],
      ["FLOW", "自動化連携", "5年目", "搬送ロボット・検査AIと人の作業分担を再設計"],
    ];
    mods.forEach(([name, jp, yr, d], i) => {
      const y = top + 1.05 + i * 0.56;
      hline(s, lx, y, lw, K.rule, T.stroke.rule);
      txt(s, [run(name, { bold: true, fontSize: 12, breakLine: true }), run(jp, { fontSize: 8.5, color: K.gray700 })], { x: lx, y: y + 0.04, w: 1.25, h: 0.48, valign: "middle" });
      txt(s, d, { x: lx + 1.3, y: y + 0.04, w: lw - 1.95, h: 0.48, size: 9.5, valign: "middle" });
      txt(s, yr, { x: lx + lw - 0.6, y: y + 0.04, w: 0.6, h: 0.48, size: 8.5, color: K.gray700, align: "right", valign: "middle" });
    });
    hline(s, lx, top + 1.05 + 4 * 0.56, lw, K.rule, T.stroke.rule);

    // ROI line chart
    const rx = 5.05, rw = 4.55;
    txt(s, "顧客側の投資回収（対象者100名の拠点・万円）", { x: rx, y: top, w: rw, h: 0.26, size: 10.5, bold: true });
    const months = Array.from({ length: 13 }, (_, m) => String(m));
    const effect = Array.from({ length: 13 }, (_, m) => m * 240);
    const cost = Array.from({ length: 13 }, () => 500);
    const axis = { catAxisLabelColor: K.gray700, valAxisLabelColor: K.gray700, catAxisLabelFontSize: 8, valAxisLabelFontSize: 8,
      catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", valGridLine: { color: K.rule, size: 0.5 }, catGridLine: { style: "none" },
      valAxisMinVal: 0, valAxisMaxVal: 3000, valAxisMajorUnit: 500, catAxisLineShow: true, valAxisLineShow: false, showLegend: false };
    s.addChart([
      { type: pres.charts.LINE, data: [{ name: "累積削減効果", labels: months, values: effect }], options: { chartColors: [K.ink], lineSize: 3, lineDataSymbol: "circle", lineDataSymbolSize: 5 } },
      { type: pres.charts.LINE, data: [{ name: "初年度費用", labels: months, values: cost }], options: { chartColors: [K.gray500], lineSize: 2, lineDash: "dash", lineDataSymbol: "none" } },
    ], { x: rx, y: top + 0.28, w: rw, h: 2.62, ...axis, objectName: "roi-chart" });
    txt(s, "← 約2.1か月で回収", { x: rx + 1.05, y: top + 2.3, w: 1.6, h: 0.22, size: 9.5, bold: true });
    txt(s, "初年度費用 500万円", { x: rx + 2.85, y: top + 2.3, w: 1.6, h: 0.22, size: 8.5, color: K.gray700 });
    txt(s, "12か月で2,880万円（5.8倍）", { x: rx + 0.65, y: top + 0.4, w: 2.4, h: 0.26, size: 10.5, bold: true });
    txt(s, "価格＝基盤100万＋登録20万×5＋モジュール50万×2。効果＝時間単価4,000円×月6h×100名＝月240万（月6hは建設業の電子化実績を準用した仮定）。横軸は経過月数", { x: rx, y: top + 2.94, w: rw, h: 0.34, size: S.note, color: K.gray700 });

    box(s, { x: X0, y: 4.8, w: W, h: 0.38, fill: K.fill });
    hline(s, X0, 4.8, W, K.ink, T.stroke.strong);
    txt(s, [
      run("価格戦略：", { bold: true }), run("価値基準価格。"),
      run("年300万円＝顧客効果2,880万円の約1割", { bold: true }),
      run("（初年度のみ導入支援200万円）"),
    ], { x: X0 + 0.15, y: 4.8, w: W - 0.3, h: 0.38, size: 10, valign: "middle" });
  }

  // ---------- Slide 5 競合分析（STP・競争戦略） ----------
  {
    const s = add(5);
    heading(s, "競合分析 ─ 既存の仕組みは置き換えず、判断を渡す層を担う", "空いているのは「判断の理由」を「仕組み」で扱う象限。ニッチャーとして差別化集中で取る");
    // positioning map
    const mx = X0, my = 1.15, mw = 4.5, mh = 3.55;
    txt(s, "ポジショニングマップ", { x: mx, y: my, w: mw, h: 0.26, size: 10.5, bold: true });
    const px = mx + 0.35, py = my + 0.42, pw = mw - 0.5, ph = mh - 0.85;
    box(s, { x: px + pw / 2, y: py, w: pw / 2, h: ph / 2, fill: K.fill, name: "target-quadrant" });
    hline(s, px, py + ph / 2, pw, K.gray500, T.stroke.rule);
    s.addShape("line", { x: px + pw / 2, y: py, w: 0, h: ph, line: { color: K.gray500, width: T.stroke.rule } });
    txt(s, "仕組み（プロダクト）", { x: px + pw / 2 - 1.0, y: py - 0.24, w: 2.0, h: 0.2, size: 8.5, color: K.gray700, align: "center" });
    txt(s, "人の時間（労働集約）", { x: px + pw / 2 - 1.0, y: py + ph + 0.04, w: 2.0, h: 0.2, size: 8.5, color: K.gray700, align: "center" });
    txt(s, "← 実績（何が起きたか）", { x: px, y: py + ph / 2 + 0.03, w: 1.9, h: 0.2, size: 8, color: K.gray700 });
    txt(s, "判断の理由（なぜ）→", { x: px + pw - 1.9, y: py + ph / 2 + 0.03, w: 1.9, h: 0.2, size: 8, color: K.gray700, align: "right" });
    const dots = [
      ["MES・ERP", 0.12, 0.16, false, "r"],
      ["搬送ロボット・検査AI", 0.1, 0.36, false, "r"],
      ["コンサル（BPR）", 0.56, 0.84, false, "r"],
      ["社内の熟練者（属人）", 0.84, 0.66, false, "b"],
      ["三現ワークス", 0.8, 0.2, true, "l"],
    ];
    dots.forEach(([label, fx, fy, ours, side]) => {
      const cx = px + fx * pw, cy = py + fy * ph, r = ours ? 0.11 : 0.08;
      s.addShape("ellipse", { x: cx - r, y: cy - r, w: 2 * r, h: 2 * r, fill: { color: ours ? K.ink : K.paper }, line: { color: K.ink, width: 1.25 }, objectName: "pos-" + label });
      const o = { size: ours ? 10 : 8.5, bold: ours, valign: "middle", h: 0.24, w: 1.6 };
      if (side === "r") txt(s, label, { ...o, x: cx + r + 0.05, y: cy - 0.12, align: "left" });
      else if (side === "l") txt(s, label, { ...o, x: cx - r - 1.65, y: cy - 0.12, align: "right" });
      else txt(s, label, { ...o, x: cx - 1.2, y: cy + r + 0.02, w: 1.4, align: "right" });
    });

    // STP + competitive strategy + SWOT
    const sx = 5.05, sw = 4.55;
    const label = (k, y, h) => {
      box(s, { x: sx, y: y + 0.03, w: 0.3, h: h - 0.06, fill: K.ink });
      txt(s, k, { x: sx, y: y + 0.03, w: 0.3, h: h - 0.06, size: 9, bold: true, color: K.paper, align: "center", valign: "middle" });
    };
    txt(s, "STP", { x: sx, y: my, w: sw, h: 0.26, size: 10.5, bold: true });
    [
      ["S", "製造業を工程の形で区分：連続工程（電線・素材）／組立／施工現場"],
      ["T", "電線・ケーブル工場。長尺・ロット・試験値が揃い、熟練者の判断が品質を左右"],
      ["P", "実績はMES・ERP、動作はロボットに任せ、判断の理由を人と機械に渡す層"],
    ].forEach(([k, v], i) => {
      const y = my + 0.28 + i * 0.38;
      label(k, y, 0.36);
      txt(s, v, { x: sx + 0.4, y, w: sw - 0.4, h: 0.36, size: 8.5, valign: "middle" });
    });
    txt(s, "競争戦略", { x: sx, y: my + 1.48, w: sw, h: 0.26, size: 10.5, bold: true });
    txt(s, [
      run("コトラーの競争地位：", { bold: true }), run("ニッチャー。大手が採算に乗せにくい専門領域に絞る", { breakLine: true }),
      run("ポーターの基本戦略：", { bold: true }), run("差別化集中。価格ではなく現場の知見で選ばれる"),
    ], { x: sx, y: my + 1.76, w: sw, h: 0.5, size: 8.5, psa: 2 });
    txt(s, "SWOT", { x: sx, y: my + 2.32, w: sw, h: 0.26, size: 10.5, bold: true });
    [
      ["S", "施工管理5年と自作開発の両立／製品が使われる現場からの逆算"],
      ["W", "製造現場の実務経験が浅い／営業・資金調達が未経験／創業者依存"],
      ["O", "担い手の高齢化による技能継承需要／フィジカルAIの実用化"],
      ["T", "大手SIer・ロボットベンダーの参入／内製化／母数が小さい"],
    ].forEach(([k, v], i) => {
      const y = my + 2.6 + i * 0.29;
      label(k, y, 0.28);
      txt(s, v, { x: sx + 0.4, y, w: sw - 0.4, h: 0.28, size: 8.5, valign: "middle" });
    });
    txt(s, "※ ERP・MES・SCMとはAPI連携で併存し、置き換えない。電線業界：353事業所・出荷額1兆8,822億円（経済構造実態調査 2024年、従業者10人以上）", { x: X0, y: 4.92, w: W, h: 0.34, size: S.note, color: K.gray700 });
  }
  // ---------- Slide 6 組織と採用 ----------
  {
    const s = add(6);
    heading(s, "設問②　組織と採用 ─ 顧客数は、導入担当の人数から逆算する", "導入担当1名あたり年6拠点。代表（CEO）は型を渡す役割に絞り、抱え込まない");
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
    heading(s, "設問⑤　経営課題 ─ 最大のリスクは、自分が律速になること", "だから役割を定義し、創業者が1か月不在でも回る状態を5年目の目標に置く");
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
        ["大手SIer・ロボットベンダーの参入", "蓄積した判断記録が障壁。機能は模倣可能、現場の履歴は移設不能"],
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
