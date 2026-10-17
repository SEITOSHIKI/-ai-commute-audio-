// カンツギ 事業計画書「16枚版」: 本編（33枚）の中身を、A4表裏（8枚×2面）の配布と10分の発表に収めたもの。
// デザインは本編と同じ。2〜3枚分を1枚に統合し、深さ（式・表・前提）は残す。数字はすべて tora_v2.js から。
// Usage: node build_tora16.js [outDir]   (needs pptxgenjs and sharp)
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const L = require("./tora_lib.js");
const V = require("./tora_v2.js");
const { estimate } = require("./tora_market.js");

const { K, X0, W, ACC, txt, box, hline, arrow, run, heading, table } = L;
const OUT_DIR = process.argv[2] || __dirname;
const OUT = path.join(OUT_DIR, "カンツギ_事業計画書_16枚版.pptx");
const ASSETS = path.join(__dirname, "assets");

// ---------- numbers ----------
const CV = V.customerValue(), U = V.unit(), PLAN = V.run(), R = PLAN.rows, Y5 = R[4];
const D15plan = { devices: 15, mix: { glass: 12, display: 3 } };
const R10 = V.run({ years: 10 }), R10_15 = V.run({ years: 10, plan: D15plan }), D15 = V.run({ plan: D15plan });
const be = Math.round(CV.breakEvenMin);
const man = (v) => Math.round(v).toLocaleString("en-US");
const mil = (v) => Math.round(v / 100);
const tri = (v) => (v < 0 ? "▲" + man(-v) : man(v));
const oku = (v) => (v >= 10000 ? Math.floor(v / 10000) + "億" + (v % 10000 ? man(v % 10000) + "万円" : "円") : man(v) + "万円");
const fund = V.GROWTH.funding.reduce((a, b) => a + b, 0) + 500;
const gapSites = Math.ceil((Y5.sga - Y5.gp) / U.gpYear);
const siteYears = R10.rows.reduce((a, r) => a + r.sites, 0);
const IMPACT = { hours: siteYears * CV.freedHours, value: siteYears * CV.docs };
const hr = (v) => man(Math.round(v * 100) * 100);
const DARK = "E3A774";
const note = (s, text) => txt(s, text, { x: X0, y: 4.98, w: W, h: 0.24, size: 7.5, color: K.gray500 });
const darkBox = (s, x, y, w, h, parts, size = 9.5) => { box(s, { x, y, w, h, fill: K.ink }); txt(s, parts, { x: x + 0.14, y, w: w - 0.28, h, size, color: K.paper, valign: "middle", psa: 2 }); };
const lab = (t, dark) => run(t, { bold: true, color: dark ? DARK : ACC });

// ---------- 台本（清 俊紀の言葉：結論を先に、数字で、自分の経験とつなげる。1分約300字） ----------
const NOTE_LIST = [null,
  ["表紙と構成", `株式会社カンツギの清俊紀です。カンツギは「勘を継ぐ」と書きます。割れた器を金で継ぐ金継ぎのように、途切れそうな現場の勘をデータで継ぐ、という意味を込めました。今日は課題の5つの設問の順に、10分でお話しします。前提は、来年、十分な自己資金で事業を立ち上げることです。最初の目標は、6か月で1拠点を立ち上げることです。`],
  ["事業概要と、選んだ理由", `結論から言うと、職長がAIグラスを掛けて作業するだけで、写真台帳と日報ができ、図面と照合し、若手の教材にもなるサービスです。お客様は関東の中堅の電気工事会社で、1拠点10台で年300万円です。この事業を選んだ理由は3つあります。1つ目は、人手不足と残業規制で、人を増やしても解決できないこと。2つ目は、映像を理解するAIと軽いAIグラスが実用レベルになったこと。3つ目は、私自身が地中送電線の現場で、ベテランの判断が手順書に残らないまま退職で消えるのを見てきたことです。`],
  ["背景と課題", `困っているのは三者です。推進するDX担当、使う施工管理者と電工、費用を出す工事部長です。数字で言うと、建設業で働く人の36.7%が55歳以上で、29歳以下は11.7%しかいません。2024年からは、残業も原則月45時間までになりました。熟練者が辞めると、若手が寸法や順番を間違えて、手戻りが増えます。手戻りは建設全体で年約1.5兆円、電気工事だけで約2,400億円と見積もっています。`],
  ["解決策とデモ", `解決策はデモで説明します。若手の作業を記録して、半導電層が図面より8mm短いことを見つけます。遠隔のベテランが「テープ基準で測り直す」と指示し、その判断も記録します。写真は手順ごとに報告書へ入るので、作業者が報告書に使う時間はゼロです。他社がすぐに真似できない理由は3つです。記録は使うほど貯まること。映像のどこが大事かは、施工経験者にしか判断できないこと。電気工事の型を先に揃えることです。`],
  ["使い方と端末", `使い方は、1班に1台、職長がAIグラスを掛けるだけです。電工全員には配りません。1拠点で10台、撮るのは作業エリアだけです。録画はAIグラスに残し、作業の区切りや「ここ撮って」という声が入った場面だけをAIに送ります。端末は記録用が8台、図面を手元に表示する用が2台です。粉じんの多い現場では産業用も選べます。AIグラスはお客様名義のリースで、手続きは当社が代行します。`],
  ["競合と差別化", `競合は2種類です。1つ目は施工管理アプリで、人が撮って入力する前提です。2つ目は日立グループで、方向はほぼ同じです。ビル保守のAI Safetyと、日立パワーソリューションズの受変電設備の点検AIがあります。ただ、日立は自社の保守契約の中でAIを使う形です。技術と規模では勝てないので、対象を中堅の電気工事会社の施工に絞り、書類と技能継承を月額で提供します。日立が同じ領域に来た場合は、提携も選択肢に入れます。`],
  ["スタッフ", `5年目に${Y5.staff}人の体制にします。中心は3つの役割です。映像AIをつくるエンジニア。電気工事の施工管理を10年以上経験し、映像のどこが大事かを判断する現場監修。そして、お客様ごとの書類の形を設定する導入コンサルタントです。この設定は外注しません。外注するとノウハウが社外に残るので、社内で設定して製品に戻します。私の役割は、最初の営業と現場監修の統括、そして判断基準をつくることです。`],
  ["経営理念", `経営理念は「現場の勘を、次の担い手へ」です。大切にすることは3つです。1つ目は、現場を主役にして、入力の手間を増やさないこと。2つ目は、記録は本人の同意を取り、誰の技術かを残すこと。3つ目は、効果を毎月数字で見せることです。特に現場の方には、これは監視ではなく、自分の技術を残す仕組みだと伝えたいと考えています。`],
  ["市場とターゲット", `市場規模は、電気工事と電気通信工事の中堅以上、約3,100社に、1拠点年300万円を掛けて約141億円です。最初は関東の中堅735社のうち、電力会社や官公庁の工事が多く、書類の量が多い約180社に絞ります。業界団体の勉強会や展示会で会い、3か月の有償試験導入から本契約に進めます。5年で約85拠点、解約を引いて${Y5.sites}拠点の計画です。`],
  ["お客様の効果", `お客様の効果は、AIグラスを掛けた班の分だけで計算しています。職長の日報が1日15分、施工管理者の写真整理が20分減ると、10台で年約${man(Math.round(CV.docs / 10) * 10)}万円です。お客様の支払いはリースを含めて約${man(CV.cost)}万円なので、約${CV.ratio.toFixed(1)}倍です。言い換えると、1台1日${be}分減れば元が取れます。この${be}分を、実証の合格ラインにします。`],
  ["当社の収益", `当社の収益は、基盤が月5万円、端末が1台月2万円で、10台なら年300万円です。AIと保存の費用は1台月約6,000円で、差し引くと1拠点で年${man(U.gpYear)}万円が残ります。粗利率は${Math.round(V.marginSite() * 100)}%です。注意点は、全ての映像をAIで解析すると1台月5万円かかり、赤字になることです。なので、解析するのは選んだ場面だけにします。`],
  ["年単位の計画と収支", `年単位の計画です。1年目は3社の試験導入、3年目は図面との照合、5年目は教材化まで進めます。5年目で${Y5.sites}拠点、売上は約${(Y5.sales / 1e4).toFixed(1)}億円です。ただし、人を先に採用するので5年間は赤字で、5年目も約${man(Math.round(-Y5.op / 100) * 100)}万円の赤字です。損益分岐点まで、あと約${gapSites}拠点分が足りません。`],
  ["資金と黒字化の条件", `資金は自己資金で約${(fund / 1e4).toFixed(2)}億円です。一度には入れず、実証に合格したら1年目に2億円、3年目に継続率と台数の基準を満たしたら5,000万円を追加します。黒字化の条件は、拠点数より1拠点あたりの台数です。15台に増やせれば、新規が計画の6割でも5年目に黒字になります。撤退ラインも決めていて、実証で1台1日15分も減らなければ撤退します。`],
  ["10年の見通し", `10年で見た場合です。5年目までの計画は変えずに延ばすと、10台のままでも${R10.firstBlack}年目に黒字になり、${R10.cumBack}年目に5年間の赤字を回収します。10年目の営業利益は約${(R10.rows[9].op / 1e4).toFixed(1)}億円です。利用料が毎年積み上がる形なので、6年目以降は利益が伸びます。条件は、6年目から採用のペースを落とすことです。値下げやAI費用が2倍になった場合は、現金が一時的に足りなくなります。`],
  ["段階ごとの課題", `段階ごとの課題です。準備期は効果が出るか、導入期は売れるか、成長期は台数と採用、成熟期は大手との競合です。計画を見直して見つけた弱いところもあります。AIグラスは録画時間と電池に制約があること。300万円という価格を、まだお客様に確認できていないこと。10台のままでは5年で黒字にならないこと。そして、私自身が律速になるリスクです。どれも確かめ方を決めています。`],
  ["準備期と、最後に", `最初の6か月は約500万円で、試作品をつくり、1拠点で実証します。最後に、この事業への思いです。10年続けば、書類に使っていた約${man(Math.round(IMPACT.hours / 1e4))}万時間が、現場の仕事と若手の指導に戻る計算です。私にとって利益は、事業を続けるための条件です。配当は求めず、共感してくれる会社や職人の方と一緒に育てたいと考えています。ただし、撤退ラインは守ります。まずは6か月で1拠点を立ち上げます。ありがとうございました。`],
];
const mmss = (s) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
const NOTES = {};
{
  let at = 0;
  NOTE_LIST.forEach((p, n) => {
    if (!p) return;
    const d = Math.max(20, Math.round(p[1].length / 5 / 5) * 5);
    NOTES[n] = `【${mmss(at)} - ${mmss(at + d)}】\n要約：${p[0]}\n\n${p[1]}`;
    at += d;
  });
  NOTES.total = mmss(at);
}

async function main() {
  const pres = new pptxgen();
  const add = await L.setup(pres, "株式会社カンツギ 事業計画書（16枚版）", NOTES);
  const yrs = ["1年目", "2年目", "3年目", "4年目", "5年目"];

  // ---------- 1 表紙・構成 ----------
  {
    const s = add(1);
    box(s, { x: 0, y: 0, w: 3.4, h: 5.625, fill: K.ink });
    s.addImage({ data: await L.png("kantsugi-logotype-reverse"), x: 0.35, y: 0.7, w: 2.7, h: 2.7 * 120 / 560 });
    txt(s, "現場の勘を、\n次の担い手へ。", { x: 0.35, y: 1.55, w: 2.9, h: 0.95, size: 20, bold: true, color: K.paper, lsm: 1.2 });
    txt(s, "株式会社カンツギ", { x: 0.35, y: 2.7, w: 2.9, h: 0.3, size: 11, color: K.paper });
    txt(s, "勘継＝金継ぎのように現場の勘を継ぐ", { x: 0.35, y: 3.0, w: 2.9, h: 0.28, size: 9.5, color: K.paper });
    txt(s, "発表者：清 俊紀", { x: 0.35, y: 4.6, w: 2.9, h: 0.3, size: 10, color: K.paper });
    const rx = 3.75, rw = 5.85;
    txt(s, "事業計画書（10分・16枚）", { x: rx, y: 0.35, w: rw, h: 0.26, size: 11, color: K.gray700 });
    txt(s, "人が減っても、現場が迷わず回る仕組みをつくる", { x: rx, y: 0.62, w: rw, h: 0.42, size: 16, bold: true });
    txt(s, "AIグラスで熟練者の作業を記録し、書類づくりと若手の育成を自動化する", { x: rx, y: 1.05, w: rw, h: 0.45, size: 10.5 });
    table(s, [["設問", "一言の答え", "頁"],
      [{ text: "① 事業・理由・想い", bold: true }, "AIグラスで作業を記録し、書類と育成を自動化する。現場で勘が消えるのを止めたい", "2〜6"],
      [{ text: "② スタッフ", bold: true }, `5年目${Y5.staff}人。現場監修と導入コンサルタントを社内に置く`, "7"],
      [{ text: "③ 経営理念", bold: true }, "現場の勘を、次の担い手へ。記録は監視ではなく、技術を残すため", "8"],
      [{ text: "④ 経営計画", bold: true }, `1拠点年300万円。5年間は赤字、10年なら${R10.firstBlack}年目に黒字。15台なら5年目`, "9〜14"],
      [{ text: "⑤ 段階ごとの課題", bold: true }, "効果 → 販売と定着 → 台数と採用 → 大手との競合", "15〜16"]],
      { x: rx, y: 1.62, w: rw, colW: [1.6, 3.75, 0.5], rowH: 0.36, size: 9.5, leftCols: [1], align: "center", name: "toc" });
    box(s, { x: rx, y: 3.95, w: rw, h: 0.9, line: ACC, lw: 1.5 });
    txt(s, [lab("課題の前提："), run("来年、豊富な資金をもとに新しい事業を立ち上げて経営する。資金は自己資金（融資・出資は使わない）。", { breakLine: true }),
      lab("最初の目標："), run(`6か月で1拠点を立ち上げる。合格ラインは1台1日${be}分の削減`)], { x: rx + 0.14, y: 3.95, w: rw - 0.28, h: 0.9, size: 9.5, valign: "middle", psa: 3 });
  }

  // ---------- 2 ①事業概要・選んだ理由と想い ----------
  {
    const s = add(2);
    heading(s, "① 事業概要：AIグラスで記録し、書類と育成を自動化する", "職長がAIグラスを掛けて作業するだけで、写真台帳・日報・図面との照合・若手の教材までつくる");
    const q = [
      ["何をするか", "職長・施工管理者がAIグラスを掛けて作業するだけで、一人称の動画と会話を記録する。AIが写真台帳・日報をつくり、図面と照合し、教材にする"],
      ["既存サービスとの違い", "施工管理アプリは人が撮って入力する前提。当社は自動で記録し、照合・報告・教材までまとめて行う。中身は施工経験者が監修する"],
      ["誰の悩みか", "関東の中堅電気工事会社。残業規制と高齢化で、写真台帳・報告書と若手の育成に回す時間がない"],
      ["収益構造", `基盤 月5万円＋端末 月2万円×台数。10台で年300万円の利用料を毎年いただく。初年度のみ導入支援200万円`],
    ];
    const cw = 2.62, ch = 1.72, top = 1.12;
    q.forEach(([h, v], i) => {
      const x = X0 + (i % 2) * (cw + 0.12), y = top + Math.floor(i / 2) * (ch + 0.12);
      box(s, { x, y, w: cw, h: ch, fill: i === 3 ? K.ink : K.fill });
      const c = i === 3 ? K.paper : K.ink;
      txt(s, h, { x: x + 0.12, y: y + 0.08, w: cw - 0.24, h: 0.3, size: 11.5, bold: true, color: i === 3 ? DARK : ACC });
      txt(s, v, { x: x + 0.12, y: y + 0.42, w: cw - 0.24, h: ch - 0.5, size: 9.5, color: c });
    });
    const rx = 6.0, rw = 3.6;
    txt(s, "選んだ理由", { x: rx, y: top, w: rw, h: 0.26, size: 11, bold: true });
    [["市場", "55歳以上が36.7%、残業は月45時間まで。人を増やして解決できない"], ["技術", "映像を理解するAIと、約50gのAIグラスが実用レベルになった"], ["自分", "地中送電線の施工管理を5年。現場とAI実装の両方がわかる"]].forEach(([k, v], i) => {
      const y = top + 0.32 + i * 0.6;
      box(s, { x: rx, y: y + 0.04, w: 0.62, h: 0.48, fill: i === 2 ? ACC : K.ink });
      txt(s, k, { x: rx, y: y + 0.04, w: 0.62, h: 0.48, size: 10.5, bold: true, color: K.paper, align: "center", valign: "middle" });
      txt(s, v, { x: rx + 0.7, y, w: rw - 0.7, h: 0.56, size: 9.5, valign: "middle" });
    });
    darkBox(s, rx, 3.2, rw, 1.2, [lab("想い", true), run("　地中送電の現場で、ベテランの判断が手順書に残らないまま退職で消えるのを見てきた。その判断をデータで残し、次の担い手に渡したい", { breakLine: false })], 10);
    note(s, "最初の顧客：関東の中堅電気工事会社（施工管理者30人以上）。成長期以降に電線・素材の工場へ広げる");
  }

  // ---------- 3 ①背景・課題・損失 ----------
  {
    const s = add(3);
    heading(s, "① 背景と課題：熟練者が辞めると、現場が止まる", "人手不足と残業規制で、書類と育成に回す時間がない。困っているのは次の三者");
    const ps = [
      ["DX推進担当", "推進する人", ["ツールを入れても現場に定着しない", "成果を数字で説明できない"]],
      ["施工管理者・電工", "使う人", ["写真の整理・日報・報告書に毎日追われる", "聞ける熟練者がいない。手順書に勘どころがない"]],
      ["工事部長・社長", "お金を出す人", ["残業に上限。人を増やせない", "熟練者の判断が退職とともに消える"]],
    ];
    const cw = 2.97, gap = 0.145, top = 1.12, ch = 1.55;
    ps.forEach(([who, sub, items], i) => {
      const x = X0 + i * (cw + gap);
      box(s, { x, y: top, w: cw, h: ch, line: K.rule });
      box(s, { x, y: top, w: cw, h: 0.42, fill: K.ink });
      txt(s, [run(who, { bold: true, fontSize: 11 }), run("　" + sub, { fontSize: 8.5 })], { x: x + 0.12, y: top, w: cw - 0.24, h: 0.42, color: K.paper, valign: "middle" });
      txt(s, items.map((v, j) => run("・" + v, { breakLine: j < items.length - 1 })), { x: x + 0.12, y: top + 0.48, w: cw - 0.24, h: ch - 0.55, size: 9.5, psa: 4 });
    });
    const sy = 2.82;
    [["36.7%", "建設業で働く人のうち55歳以上（29歳以下は11.7%）"], ["月45時間", "2024年4月から建設業にも残業の上限（原則）"], ["年約1.5兆円", "建設全体の手戻り（建設投資75.6兆円×2%）"], ["約2,400億円", "うち電気工事（完成工事高 約12兆円×2%）"]].forEach(([f, l], i) => {
      const cw2 = (W - 0.3) / 4, x = X0 + i * (cw2 + 0.1);
      box(s, { x, y: sy, w: cw2, h: 1.12, fill: i >= 2 ? K.ink : K.fill });
      txt(s, f, { x: x + 0.1, y: sy + 0.06, w: cw2 - 0.2, h: 0.48, size: 19, bold: true, color: i >= 2 ? DARK : K.ink, valign: "middle" });
      txt(s, l, { x: x + 0.1, y: sy + 0.56, w: cw2 - 0.2, h: 0.52, size: 8.5, color: i >= 2 ? K.paper : K.ink });
    });
    box(s, { x: X0, y: 4.1, w: W, h: 0.78, line: ACC, lw: 1.5 });
    txt(s, [lab("熟練者が辞めた後に起きること："), run("若手が寸法・基準点・重なり順を間違える → 手戻りと工期遅れが出る → 残業規制で人を足せず、受けられる工事が減る")], { x: X0 + 0.14, y: 4.1, w: W - 0.28, h: 0.78, size: 10, valign: "middle" });
    note(s, "国土交通省（労働力調査 2024年平均、477万人）、労働基準法。手戻り率2%は海外研究（工事費の0.4〜6%）から控えめに置いた仮定");
  }

  // ---------- 4 ①解決策・デモ ----------
  {
    const s = add(4);
    heading(s, "① 解決策：作業を自動で記録し、3つの形で渡す", "デモ（架空のケーブル接続工事）：若手の作業を記録し、図面との8mmの差を見つけ、報告書と教材にする");
    const lw = 5.0, top = 1.1, ih = lw / 2.382;
    s.addImage({ path: path.join(ASSETS, "shot-glass.png"), x: X0, y: top, w: lw, h: ih });
    [["検出", "半導電層の端部 実測112mm／図面120mm（8mm不足）"], ["是正", "遠隔の熟練者が「テープ基準で測り直す」と指示。判断として記録"], ["報告", "手順ごとに写真を自動割当。作業者の報告書作成は0分"]].forEach(([k, v], i) => {
      const y = top + ih + 0.1 + i * 0.32;
      box(s, { x: X0, y: y + 0.03, w: 0.5, h: 0.25, fill: K.ink });
      txt(s, k, { x: X0, y: y + 0.03, w: 0.5, h: 0.25, size: 9, bold: true, color: K.paper, align: "center", valign: "middle" });
      txt(s, v, { x: X0 + 0.6, y, w: lw - 0.6, h: 0.31, size: 9.5, valign: "middle" });
    });
    const rx = 5.6, rw = 4.0;
    txt(s, "渡す先は3つ", { x: rx, y: top, w: rw, h: 0.26, size: 11, bold: true });
    [["若手へ", "図面・手順を現物に重ね、差異をその場で表示"], ["管理者へ", "写真を抜き出し、当日の写真台帳・日報を自動作成"], ["次の担い手へ", "ベテランの判断を、動画・音声・研修資料の教材に"]].forEach(([k, v], i) => {
      const y = top + 0.3 + i * 0.42;
      hline(s, rx, y, rw, K.rule, 0.75);
      txt(s, [run(k + "　", { bold: true, color: ACC }), run(v)], { x: rx, y: y + 0.02, w: rw, h: 0.4, size: 9.5, valign: "middle" });
    });
    box(s, { x: rx, y: 2.65, w: rw, h: 2.22, fill: K.ink });
    txt(s, [lab("他社がすぐに真似できない理由（3つ）", true), run("", { breakLine: true }),
      run("1. 記録は使うほど貯まる", { bold: true, breakLine: true }), run("図面に残らない現場処置の記録は、後から来た会社に移せない", { fontSize: 8.5, breakLine: true }),
      run("2. 大事な場面の判断は施工経験者にしかできない", { bold: true, breakLine: true }), run("現場監修が映像に印を付け、その印でAIを学習させる", { fontSize: 8.5, breakLine: true }),
      run("3. 電気工事の型を先に揃える", { bold: true, breakLine: true }), run("工種ごとの手順と、発注者ごとの報告書の書式", { fontSize: 8.5 })],
      { x: rx + 0.14, y: 2.7, w: rw - 0.28, h: 2.12, size: 9.5, color: K.paper, valign: "middle", psa: 2 });
    note(s, "デモアプリは claude.ai 上で再生できる（作業の再生・1件ずつ送り・全体像）。寸法・人物はすべて架空");
  }

  // ---------- 5 ①使い方・端末 ----------
  {
    const s = add(5);
    heading(s, "① 使い方と端末：1班に1台、職長がAIグラスを掛ける", "端末は作らず市販品を使う。誰が・何台・どこで・何を撮るかを最初に決める");
    const top = 1.12, lw = 4.8;
    table(s, [["", "決めごと（1拠点の例）"],
      ["誰が", "1班に1台、職長が掛ける（電工全員には配らない）"],
      ["何台", "1拠点10台＝5現場×2班。効果が出た班から増やす"],
      ["どこで", "作業エリアだけ。休憩・移動・私語は記録しない"],
      ["何を", "作業の区切り・声の合図・検査の場面。録画は端末に残し、AIに送るのは選んだ場面だけ"],
      ["どう送る", "グラス → スマホ → クラウド。圏外は後で送る"],
      ["誰が見る", "施工管理者は報告書を確認、熟練者は判断を足す、若手は教材で学ぶ"]],
      { x: X0, y: top, w: lw, colW: [0.85, 3.95], rowH: [0.26, 0.34, 0.34, 0.34, 0.48, 0.34, 0.48], size: 9.5, leftCols: [1], name: "usage" });
    const rx = 5.45, rw = 4.15;
    txt(s, "端末の構成（標準10台）", { x: rx, y: top, w: rw, h: 0.26, size: 11, bold: true });
    [["8台", "カメラ付きAIグラス", "約8万円・約50g。記録と音声メモ", ACC], ["2台", "表示付きARグラス", "約12万円・約70g。手元に図面と手順", K.ink], ["選択", "産業用ヘッドセット", "約37万円〜。粉じん・高所の現場だけ", K.gray500]].forEach(([n, a, b, c], i) => {
      const y = top + 0.3 + i * 0.56;
      box(s, { x: rx, y, w: 0.6, h: 0.5, fill: c });
      txt(s, n, { x: rx, y, w: 0.6, h: 0.5, size: 11, bold: true, color: K.paper, align: "center", valign: "middle" });
      txt(s, [run(a, { bold: true, breakLine: true }), run(b, { fontSize: 8.5, color: K.gray700 })], { x: rx + 0.68, y, w: rw - 0.68, h: 0.5, size: 10, valign: "middle" });
    });
    box(s, { x: rx, y: 3.05, w: rw, h: 0.85, fill: K.fill });
    txt(s, [lab("手元AR："), run("壁に貼った約2mの組立図を、手元に実物大で。ミスしやすい所は音声とレーザーで止める。寸法の最終確認はスケール")], { x: rx + 0.12, y: 3.05, w: rw - 0.24, h: 0.85, size: 9.5, valign: "middle" });
    darkBox(s, X0, 4.05, W, 0.82, [lab("グラスの持ち方：", true), run(`試験導入は当社がレンタル → 本契約はお客様名義のリース（手続きは当社が代行、10台で年約${man(V.leaseSite())}万円）→ 3年目から端末込みの料金も選べる`, { breakLine: true }),
      lab("当社がつくるもの：", true), run("アプリ・AI・データ基盤。AIグラス本体はつくらず、性能が上がったら乗り換える")], 9.5);
    note(s, "価格・重さは公表値と販売店の参考価格。国内の正規販売・技適・保護めがねの規格・電池の持ちは準備期に実機で確認する");
  }

  // ---------- 6 ①競合・差別化 ----------
  {
    const s = add(6);
    heading(s, "① 競合と差別化：日立と方向は同じ。対象を施工と中堅に絞る", "施工管理アプリは人が入力する前提。日立は自社の保守契約にAIを組み込む。当社は中堅の施工を月額で");
    const mx = X0, my = 1.12, mw = 4.3, mh = 3.75;
    txt(s, "ポジショニングマップ", { x: mx, y: my, w: mw, h: 0.26, size: 11, bold: true });
    const px = mx + 0.25, py = my + 0.5, pw = mw - 0.35, ph = mh - 1.0;
    box(s, { x: px + pw / 2, y: py, w: pw / 2, h: ph / 2, fill: K.fill });
    hline(s, px, py + ph / 2, pw, K.gray500, 0.75);
    s.addShape("line", { x: px + pw / 2, y: py, w: 0, h: ph, line: { color: K.gray500, width: 0.75 } });
    txt(s, "判断を渡す・教える", { x: px + pw / 2 - 1.0, y: py - 0.24, w: 2.0, h: 0.2, size: 8.5, color: K.gray700, align: "center" });
    txt(s, "記録して終わり", { x: px + pw / 2 - 1.0, y: py + ph + 0.03, w: 2.0, h: 0.2, size: 8.5, color: K.gray700, align: "center" });
    txt(s, "← 人が撮って入力", { x: px, y: py + ph / 2 + 0.03, w: 1.4, h: 0.2, size: 8, color: K.gray700 });
    txt(s, "自動で記録 →", { x: px + pw - 1.4, y: py + ph / 2 + 0.03, w: 1.4, h: 0.2, size: 8, color: K.gray700, align: "right" });
    [["ANDPAD（施工管理）", 0.12, 0.66, false, "r"], ["SPIDERPLUS", 0.3, 0.56, false, "r"], ["蔵衛門（工事写真）", 0.08, 0.86, false, "r"], ["tebiki（手順）", 0.14, 0.32, false, "r"],
      ["日立PS（受変電の点検）", 0.44, 0.14, false, "l"], ["遠隔臨場カメラ", 0.8, 0.88, false, "l"], ["RealWear", 0.7, 0.42, false, "l"], ["日立 AI Safety", 0.82, 0.3, false, "l"], ["カンツギ", 0.9, 0.16, true, "l"]].forEach(([label, fx, fy, ours, side]) => {
      const cx = px + fx * pw, cy = py + fy * ph, r = ours ? 0.11 : 0.07;
      s.addShape("ellipse", { x: cx - r, y: cy - r, w: 2 * r, h: 2 * r, fill: { color: ours ? K.ink : K.paper }, line: { color: K.ink, width: 1.25 } });
      const o = { size: ours ? 10 : 8, bold: ours, valign: "middle", h: 0.22, w: 1.7 };
      if (side === "r") txt(s, label, { ...o, x: cx + r + 0.04, y: cy - 0.11, align: "left" });
      else txt(s, label, { ...o, x: cx - r - 1.74, y: cy - 0.11, align: "right" });
    });
    const rx = 4.95, rw = 4.65;
    table(s, [["", "日立 AI Safety", "日立PS 点検AI", "カンツギ"],
      ["仕事", "ビル設備の保守", "受変電設備の点検", { text: "電気工事の施工", bold: true }],
      ["使う人", "日立の自社技術者", "日立PSの保守員", { text: "中堅の電気工事会社", bold: true }],
      ["AIの働き", "映像で危険を通知", "点検前後の写真でミスを照合", { text: "記録→書類→教材", bold: true }],
      ["稼ぎ方", "保守契約・ビル向け", "保守契約（AIは品質と原価）", { text: "月額の利用料", bold: true }],
      ["段階", "現場で運用中", "検証中", { text: "準備期", bold: true }]],
      { x: rx, y: my, w: rw, colW: [0.75, 1.25, 1.35, 1.3], rowH: 0.36, size: 8.5, leftCols: [1, 2, 3], align: "left", name: "vs" });
    darkBox(s, rx, 3.4, rw, 1.47, [lab("差別化の方針", true), run("", { breakLine: true }),
      run("① 安全・保守の領域では競わない", { breakLine: true }), run("② 施工の書類（写真台帳・検査記録）を発注者の書式で出す", { breakLine: true }),
      run("③ 電気工事の工種ごとの型と教材を先に揃える", { breakLine: true }), lab("日立が同じ領域に来た場合：", true), run("提携も選択肢に入れる。2〜3年で型とデータを先に揃える")], 9);
    note(s, "日立PS＝日立パワーソリューションズ。出典：日立ニュースリリース（2025年3月・10月・11月）、Physical AI 事例（2026年7月）。日立の価格・外販は未確認");
  }

  // ---------- 7 ②スタッフ ----------
  {
    const s = add(7);
    heading(s, `② スタッフ：5年目${Y5.staff}人。現場監修とコンサルを社内に置く`, "役割ごとの人数・求める特性・私の役割。各社の帳票の設定は外注せず、ノウハウを社内に貯める");
    const lx = X0, lw = 2.85, top = 1.12;
    box(s, { x: lx, y: top, w: lw, h: 3.75, fill: K.ink });
    txt(s, [run("代表取締役 CEO", { fontSize: 9, breakLine: true }), run("代表（創業者）", { bold: true, fontSize: 13 })], { x: lx + 0.14, y: top + 0.08, w: lw - 0.28, h: 0.55, color: K.paper });
    txt(s, [run("電線メーカーで66kV／275kV 地中送電線の施工管理5年", { breakLine: true }), run("Python・FastAPI・Docker・RAGを独学し業務に実装", { breakLine: true }),
      run("ニュース要約→音声→毎朝配信を個人で運用中", { breakLine: true }), run("1級電気工事施工管理技士・G検定", { breakLine: true }),
      run("私の役割：最初の営業・現場監修の統括・判断基準づくり", { bold: true, color: DARK })], { x: lx + 0.14, y: top + 0.72, w: lw - 0.28, h: 2.95, size: 9.5, color: K.paper, psa: 6 });
    const rx = 3.45, rw = 6.15;
    table(s, [["役割", "人数 1→5年目", "任せる仕事", "求める特性"],
      ...V.ROLES.slice(1).map((r) => [r[0].replace(/（.*?）/, ""), `${r[2][0]}→${r[2][4]}人`, r[3].replace(/（当社のコンサル部隊）/, ""), r[4]]),
      ["社外顧問", "2人", "映像AI（大学）・電気工事業界のOB", "月1〜2回の助言"]],
      { x: rx, y: top, w: rw, colW: [1.25, 0.75, 2.35, 1.8], rowH: 0.33, size: 8.5, leftCols: [2, 3], name: "team" });
    box(s, { x: rx, y: 4.2, w: rw, h: 0.67, fill: K.fill });
    txt(s, [lab("導入コンサルタントを社内に置く理由："), run("各社の帳票や連携の設定を外注すると、ノウハウが社外に残る。社内で設定し、型として製品に戻す")], { x: rx + 0.12, y: 4.2, w: rw - 0.24, h: 0.67, size: 9.5, valign: "middle" });
    note(s, "人件費は役割別の年収×1.2（会社負担）＋採用費（年収30%×紹介会社経由50%）＋社外顧問 年180万円。パートナー：グラスのメーカー・リース会社・大学");
  }

  // ---------- 8 ③経営理念 ----------
  {
    const s = add(8);
    heading(s, "③ 経営理念：現場の勘を、次の担い手へ。", "ホームページで伝える内容：使命・大切にすること・それぞれへの約束");
    const top = 1.12, lw = 3.5;
    box(s, { x: X0, y: top, w: lw, h: 3.75, fill: K.ink });
    txt(s, [run("使命（ミッション）", { bold: true, color: DARK, fontSize: 10, breakLine: true }), run("現場の勘を、", { bold: true, fontSize: 20, breakLine: true }), run("次の担い手へ。", { bold: true, fontSize: 20, breakLine: true }),
      run("熟練者の判断をデータで継ぎ、人が減っても日本の現場の品質を守る", { fontSize: 10.5, breakLine: true }), run(" ", { fontSize: 6, breakLine: true }),
      run("目指す姿（ビジョン）", { bold: true, color: DARK, fontSize: 10, breakLine: true }), run("人が減っても、現場が迷わず回る。若手が「聞ける先輩がいない」と困らない", { fontSize: 10.5 })],
      { x: X0 + 0.18, y: top + 0.1, w: lw - 0.36, h: 3.55, color: K.paper, valign: "middle", psa: 3 });
    const rx = X0 + lw + 0.25, rw = W - lw - 0.25;
    txt(s, "大切にする3つのこと（バリュー）", { x: rx, y: top, w: rw, h: 0.26, size: 11, bold: true });
    [["現場を主役に", "作業の手を止めない。入力を増やさない。現場の言葉で話す"], ["勘に敬意を", "記録は本人の同意のもとで。誰の技術かを残し、評価につなげる"], ["数字で約束する", "効果は毎月測って見せる。効かなければ正直に言う"]].forEach(([h, d], i) => {
      const y = top + 0.32 + i * 0.52;
      box(s, { x: rx, y: y + 0.04, w: 0.38, h: 0.42, fill: ACC });
      txt(s, String(i + 1), { x: rx, y: y + 0.04, w: 0.38, h: 0.42, size: 12, bold: true, color: K.paper, align: "center", valign: "middle" });
      txt(s, [run(h, { bold: true, breakLine: true }), run(d, { fontSize: 9, color: K.gray700 })], { x: rx + 0.47, y, w: rw - 0.47, h: 0.5, size: 10.5, valign: "middle" });
    });
    txt(s, "一人ひとりへの約束（ホームページの3つの入口）", { x: rx, y: top + 1.95, w: rw, h: 0.26, size: 11, bold: true });
    table(s, [["誰に", "伝えること"], ["お客様（工事会社）", "書類の時間を減らし、効果を毎月の数字で示す"], ["現場で働く人", "監視ではない。あなたの技術を残し、若手と評価につなげる"], ["社会・次の世代", "現場の勘を業界の財産にし、AIとロボットに教える"]],
      { x: rx, y: top + 2.25, w: rw, colW: [1.6, rw - 1.6], rowH: 0.36, size: 9.5, leftCols: [1], name: "promise" });
    note(s, "トップページの一文：「ベテランの『ここが大事』を、メガネ型のカメラとAIで残し、報告書と若手の教材に変えます」");
  }

  // ---------- 9 ④市場・ターゲット・売り方 ----------
  {
    const s = add(9);
    heading(s, "④ 市場とターゲット：約141億円。最初は関東の中堅に絞る", "全体市場（年7〜9%成長）→ ターゲット市場（顧客数×年額）→ 一次ターゲット180社 → 5年で81拠点");
    const top = 1.12, lw = 4.6;
    txt(s, "ターゲット市場の数え方（顧客数×年額、億円）", { x: X0, y: top, w: lw, h: 0.26, size: 11, bold: true });
    table(s, [["業種＼規模", "大（300人以上）", "中（30〜299人）"],
      ["電気工事", "300社×1,500万＝45", { text: "2,100社×300万＝63", bold: true, fill: "F6E7DA" }],
      ["電気通信工事", "100社×1,500万＝15", "600社×300万＝18"],
      [{ text: "合計（中堅以上 約3,100社）", bold: true }, "", { text: "約141億円", bold: true, color: ACC }]],
      { x: X0, y: top + 0.3, w: lw, colW: [1.6, 1.5, 1.5], rowH: 0.32, size: 9, align: "center", name: "seg" });
    darkBox(s, X0, top + 1.72, lw, 1.15, [lab("全体市場の伸び", true), run("", { breakLine: true }),
      run("国内建設テック（建築）1,845→3,043億円、年7.4%（2023→2030年度）", { breakLine: true }),
      run(`自社が狙う市場：5年目${Y5.sites}拠点×300万円＝約${(Y5.sites * 300 / 1e4).toFixed(1)}億円（約${(Y5.sites * 300 / 141e2).toFixed(1)}%）`)], 9);
    txt(s, "小規模は書類が少なく元が取れにくいので外す。大手は支店5拠点分", { x: X0, y: top + 2.93, w: lw, h: 0.3, size: 9, color: K.gray700 });
    const rx = 5.25, rw = 4.35;
    txt(s, "一次ターゲットの4条件", { x: rx, y: top, w: rw, h: 0.26, size: 11, bold: true });
    table(s, [["規模", "施工管理者30人以上"], ["書類の重さ", "電力会社・官公庁の工事が多い"], ["手戻りの痛さ", "ケーブル接続など高くつく工種"], ["デジタルの素地", "施工管理アプリ・DX担当がいる"]],
      { x: rx, y: top + 0.3, w: rw, colW: [1.3, 3.05], rowH: 0.27, size: 9, leftCols: [1], name: "icp" });
    txt(s, "5年間の積み上げ（仮説）", { x: rx, y: top + 1.5, w: rw, h: 0.26, size: 11, bold: true });
    [["関東の中堅 735社", 1.0, K.gray500], ["4条件に合う 180社", 0.82, K.gray700], ["会える（60%）108社", 0.64, K.gray700], ["本契約（30%）32社", 0.46, ACC]].forEach(([a, k, c], i) => {
      const w = rw * k, x = rx + (rw - w) / 2, y = top + 1.8 + i * 0.32;
      box(s, { x, y, w, h: 0.28, fill: c });
      txt(s, a, { x, y, w, h: 0.28, size: 9, bold: true, color: K.paper, align: "center", valign: "middle" });
    });
    txt(s, `＋その他555社から29社 → 約85拠点（解約後${Y5.sites}）`, { x: rx, y: top + 3.08, w: rw, h: 0.24, size: 8.5, bold: true, align: "center" });
    darkBox(s, X0, 4.45, W, 0.42, [lab("売り方：", true), run("業界団体の勉強会・電設の展示会・グラスの販売店 → デモ → 3か月の有償試験導入（導入支援200万円）→ 本契約 → 協力会社へ")], 9.5);
    note(s, "電気工事業の許可業者65,497社（令和7年3月末）から規模別に推計（内訳は仮定）。矢野経済研究所 No.3789。率はすべて仮定で最初の3社で実測する");
  }

  // ---------- 10 ④お客様の効果 ----------
  {
    const s = add(10);
    heading(s, `④ お客様の効果：1拠点10台で年約${man(Math.round(CV.docs / 10) * 10)}万円。支払いの約${CV.ratio.toFixed(1)}倍`, "効果はAIグラスを掛けた班の分だけ数える。計画に入れるのは書類と写真整理の時間だけ");
    const top = 1.12, lw = 5.4;
    box(s, { x: X0, y: top, w: lw, h: 1.05, fill: K.fill });
    txt(s, [lab("式："), run(`10台 × 年240日 ×（職長15分×${hr(V.H.electrician)}円 ＋ 施工管理者20分×${hr(V.H.manager)}円）`, { breakLine: true }), run(`＝ 年 約${man(CV.docs)}万円`, { bold: true, fontSize: 18, color: ACC })],
      { x: X0 + 0.14, y: top, w: lw - 0.28, h: 1.05, size: 10, valign: "middle", psa: 3 });
    txt(s, `時給＝統計の年収×会社負担${V.WAGE.employerLoad}÷残業を含む年${man(V.WAGE.annualHours)}時間（毎月勤労統計 令和6年）。1台1日35分は仮説で、実証で測る`, { x: X0, y: top + 1.1, w: lw, h: 0.4, size: 8.5, color: K.gray700 });
    const bars = [["効果（書類の時間）", CV.docs, ACC], ["お客様の支払い", CV.cost, K.ink], ["手戻り削減を含む場合", CV.docs + CV.rework, K.gray500]];
    const sc = (lw - 2.85) / (CV.docs + CV.rework);
    bars.forEach(([n, v, c], i) => {
      const y = top + 1.55 + i * 0.44;
      txt(s, n, { x: X0, y, w: 1.85, h: 0.36, size: 9.5, bold: true, valign: "middle" });
      box(s, { x: X0 + 1.9, y: y + 0.06, w: v * sc, h: 0.24, fill: c });
      txt(s, man(v) + "万円", { x: X0 + 1.95 + v * sc, y, w: 0.9, h: 0.36, size: 10, bold: true, valign: "middle" });
    });
    txt(s, `支払い＝利用料300万円＋AIグラスのリース約${man(V.leaseSite())}万円。手戻り＝完成工事高30億円×2%×対象工種30%×削減20%（上振れ・計画外）`, { x: X0, y: top + 2.9, w: lw, h: 0.4, size: 8.5, color: K.gray700 });
    const rx = 6.1, rw = 3.5;
    box(s, { x: rx, y: top, w: rw, h: 1.45, fill: K.ink });
    txt(s, [lab("元が取れる条件", true), run("", { breakLine: true }), run(`1台1日 ${be}分`, { bold: true, fontSize: 22, breakLine: true }), run("想定は35分。実証の合格ラインにする", { fontSize: 9 })], { x: rx + 0.14, y: top, w: rw - 0.28, h: 1.45, size: 10, color: K.paper, valign: "middle", psa: 3 });
    const fy = CV.cost + V.PLAN.setup;
    table(s, [["1拠点10台", "万円"], ["初年度の支払い（導入支援200を含む）", man(fy)], ["回収期間（初年度）", `約${Math.round(fy / (CV.docs / 12))}か月`], ["2年目以降の差（毎年）", `＋${man(CV.docs - CV.cost)}`], ["手戻り削減を含む倍率", `約${CV.ratioUp.toFixed(1)}倍`]],
      { x: rx, y: top + 1.58, w: rw, colW: [2.4, 1.1], rowH: 0.32, size: 9.5, name: "roi" });
    box(s, { x: X0, y: 4.45, w: W, h: 0.42, line: ACC, lw: 1.5 });
    txt(s, [lab("新人の指導："), run("他の業務と切り分けられないので金額にはしない。独り立ちまでの月数・付き添い時間・やり直し件数で測る")], { x: X0 + 0.14, y: 4.45, w: W - 0.28, h: 0.42, size: 9.5, valign: "middle" });
    note(s, "年収：賃金構造基本統計調査（電気工事従事者 約551万円、施工管理者は建築・土木技術者で代用 約620万円）。計算の詳細は計算詳細版 p.5〜8");
  }

  // ---------- 11 ④当社の稼ぎ方 ----------
  {
    const s = add(11);
    heading(s, `④ 当社の収益：1拠点の粗利は年${man(U.gpYear)}万円`, "基盤＋端末の台数で課金。報告書の書式に整えるところまで利用料に含む。AIグラスはお客様のリース");
    const top = 1.12, lw = 4.6;
    table(s, [["項目", "値段", "中身"],
      ["基盤", "月5万円／拠点", "記録の保管庫・図面と手順・各社の報告書の型・管理画面・月次の効果レポート"],
      ["端末ライセンス", "月2万円／台", "記録→写真の自動抽出→写真台帳・日報を各社の書式で"],
      ["導入支援", "200万円（初年度）", "導入コンサルタントが約2か月・42人日、現場に入る"],
      ["個社設定", "個別（売上の5%以下）", "大手の基幹システム連携など、標準で足りない所だけ"]],
      { x: X0, y: top, w: lw, colW: [1.05, 1.15, 2.4], rowH: [0.26, 0.5, 0.42, 0.42, 0.42], size: 8.5, leftCols: [1, 2], name: "price" });
    darkBox(s, X0, top + 2.15, lw, 1.55, [lab("1拠点（10台）の損得", true), run("", { breakLine: true }),
      run(`利用料 300万円 − AI・保存 ${man(V.cogsSite())}万円 ＝ 粗利 ${man(U.gpYear)}万円（${Math.round(V.marginSite() * 100)}%）`, { breakLine: true }),
      run(`導入支援 200万円 − 原価 約141万円 ＝ 粗利 約${man(U.setupGp)}万円`, { breakLine: true }),
      run(`LTV 約${man(U.ltv(7))}万円（7年）÷ CAC 約${man(U.cac)}万円 ＝ 約${Math.round(U.ltv(7) / U.cac)}倍`, { bold: true })], 9.5);
    const rx = 5.25, rw = 4.35, C = V.CLOUD;
    txt(s, "AI・保存の費用（端末1台・月）", { x: rx, y: top, w: rw, h: 0.26, size: 11, bold: true });
    table(s, [["中身", "円／月"], ["音声の文字起こし（1日4時間）", man(C.transcribe)], ["大事な場面の画像解析", man(C.vision)], ["日報・報告書の文章生成", man(C.report)], ["動画の保存（3年）", man(C.storage)], ["転送・監視・端末管理", man(C.ops)], [{ text: "合計", bold: true }, { text: man(V.cloudPerDevice / 12 * 10000), bold: true }]],
      { x: rx, y: top + 0.3, w: rw, colW: [3.1, 1.25], rowH: 0.27, size: 9, strongRows: [6], name: "cloud" });
    box(s, { x: rx, y: top + 2.32, w: rw, h: 1.38, line: ACC, lw: 1.5 });
    txt(s, [lab("注意点：粗利率を崩すのは台数ではなく使い方", false), run("", { breakLine: true }), run("全ての映像をクラウドで毎秒解析すると1台月約5万円かかり、価格を超える。録画は端末に残し、解析は選んだ場面だけにする。場面の選び方は、ルール → 汎用AI → 現場監修の印で学習した自社AIの3段階", { fontSize: 9 })],
      { x: rx + 0.12, y: top + 2.32, w: rw - 0.24, h: 1.38, size: 10, valign: "middle", psa: 2 });
    note(s, "単価はクラウド・AIの公表価格（$1＝150円）からの見積。CAC＝（営業3人の人件費＋広告500万円）÷新規33拠点。1〜2年目の創業者の営業時間は入れていない");
  }

  // ---------- 12 ④年単位の計画と収支 ----------
  {
    const s = add(12);
    heading(s, `④ 年単位の計画と収支：5年間は赤字。5年目は分岐点の${Math.round(Y5.sales / Y5.bep * 100)}%`, "人を先に採用するため5年間は赤字。黒字化の条件はp.13、10年の見通しはp.14");
    const top = 1.12;
    table(s, [["", "準備期", ...yrs],
      ["やること", "試作品・1拠点で実証", "試験導入3社。記録と報告を自動に", "本契約へ。写真台帳を各社の形式で", "図面との照合。大手1社を事例に", "表示付きグラスで手元AR", "教材化。電線工場へ試験"],
      ["新規／期末の拠点", "─", ...R.map((r) => `${r.nw}／${r.sites}`)],
      ["社員", "1", ...R.map((r) => String(r.staff))],
      ["売上（百万円）", "─", ...R.map((r) => String(mil(r.sales)))],
      ["粗利率", "─", ...R.map((r) => Math.round(r.gpRate * 100) + "%")],
      ["販管費（百万円）", "─", ...R.map((r) => String(mil(r.sga)))],
      [{ text: "営業損益（百万円）", bold: true }, "▲5", ...R.map((r) => ({ text: (r.op < 0 ? "▲" : "") + Math.abs(mil(r.op)), color: r.op < 0 ? ACC : K.ink, bold: true }))],
      ["期末の現金（百万円）", "─", ...R.map((r) => String(mil(r.cash)))],
      ["進む条件", `1台1日${be}分以上`, "2社以上が本契約", "平均12台", "継続率90%", "─", "─"]],
      { x: X0, y: top, w: W, colW: [1.55, 1.2, 1.29, 1.29, 1.29, 1.29, 1.29], rowH: [0.26, 0.52, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.32], size: 8.5, align: "center", leftCols: [0], strongRows: [7], name: "yearly" });
    const by = 4.15;
    box(s, { x: X0, y: by, w: 4.5, h: 0.73, fill: K.fill });
    txt(s, [lab("売上で見た損益分岐点："), run(`5年目の分岐点 約${mil(Y5.bep)}百万円に対し売上 約${mil(Y5.sales)}百万円。足りないのは約${gapSites}拠点分の粗利`)], { x: X0 + 0.12, y: by, w: 4.26, h: 0.73, size: 9.5, valign: "middle" });
    box(s, { x: 5.1, y: by, w: 4.5, h: 0.73, fill: K.ink });
    txt(s, [lab("人で見た損益分岐点：", true), run(`社員1人の販管費 約${man(Y5.sga / Y5.staff)}万円÷1拠点の粗利${man(U.gpYear)}万円＝約${(Y5.sga / Y5.staff / U.gpYear).toFixed(1)}。期末で1人約4拠点が目安、5年目は${Y5.sitesPerStaff.toFixed(2)}`)], { x: 5.22, y: by, w: 4.26, h: 0.73, size: 9.5, color: K.paper, valign: "middle" });
    note(s, "売上＝既存の利用料＋新規の利用料（半年分）＋導入支援＋個社設定。解約は年10%。人件費は役割別の年収×1.2、AIの学習費（2年目〜）も入れた簡易計算（税・償却を除く）");
  }

  // ---------- 13 ④資金・黒字化の条件 ----------
  {
    const s = add(13);
    heading(s, "④ 資金と黒字化の条件：黒字化は1拠点の台数で決まる", `自己資金${(fund / 1e4).toFixed(2)}億円を段階的に投入する。進む・直す・止まるの判断基準を先に決める`);
    const top = 1.12, lw = 4.45;
    txt(s, "資金の入れ方（すべて自己資金）", { x: X0, y: top, w: lw, h: 0.26, size: 11, bold: true });
    table(s, [["時期", "入れる条件", "金額"], ["準備期", "─", "500万円"], ["1年目", `実証で1台1日${be}分以上`, oku(V.GROWTH.funding[0])], ["3年目", "継続率90%・平均12台", oku(V.GROWTH.funding[2])], [{ text: "合計", bold: true }, "", { text: oku(fund), bold: true }]],
      { x: X0, y: top + 0.3, w: lw, colW: [0.8, 2.25, 1.4], rowH: 0.29, size: 9, leftCols: [1], strongRows: [4], name: "fund" });
    txt(s, `内訳：5年間の赤字の累計ピーク約${(-PLAN.cumPeak / 1e4).toFixed(2)}億円＋最低残高（固定費3か月分）＋遅れへの備え。現金の谷は約${mil(PLAN.minCash)}百万円`, { x: X0, y: top + 1.82, w: lw, h: 0.45, size: 8.5, color: K.gray700 });
    const sc = (k, plan) => V.run({ plan, newSites: V.GROWTH.newSites.map((n) => Math.round(n * k)), contract: V.GROWTH.contract.map((c) => c * k), hireK: k < 1 ? k : undefined });
    const yr = (r) => (r.firstBlack ? r.firstBlack + "年目" : "5年でしない");
    txt(s, "新規が計画に届かなかったら（採用も連動）", { x: X0, y: top + 2.32, w: lw, h: 0.26, size: 11, bold: true });
    table(s, [["新規の達成度", "10台のまま", "15台に増やせたら"], ...[1, 0.8, 0.6].map((k) => { const a = sc(k), b = sc(k, D15plan); return [k === 1 ? "計画どおり" : `計画の${k * 10}割`, { text: yr(a), color: a.firstBlack ? K.ink : ACC, bold: !a.firstBlack }, { text: yr(b), bold: true }]; })],
      { x: X0, y: top + 2.62, w: lw, colW: [1.45, 1.4, 1.6], rowH: 0.27, size: 9, align: "center", name: "short" });
    const rx = 5.1, rw = 4.5;
    txt(s, "進む・直す・止まるの判断基準", { x: rx, y: top, w: rw, h: 0.26, size: 11, bold: true });
    table(s, [["時期", "進む", "直す・止まる"],
      ["実証（準備期）", `1台1日${be}分以上`, `15〜${be}分：写真台帳に絞る／15分未満：撤退`],
      ["1年目末", "試験導入3社のうち2社以上が本契約", "1社：値段と顧客を見直す／0社：撤退か方向転換"],
      ["2年目末", "5拠点以上・平均12台", "未達：採用を止める"],
      ["3年目", "継続率90%以上", "未達：追加資金と採用を絞る"]],
      { x: rx, y: top + 0.3, w: rw, colW: [1.05, 1.5, 1.95], rowH: [0.26, 0.5, 0.5, 0.36, 0.36], size: 8.5, leftCols: [1, 2], name: "gates" });
    darkBox(s, rx, 3.5, rw, 1.37, [lab("黒字化の条件は、拠点数より1拠点の台数", true), run("", { breakLine: true }),
      run(`10台のままでは計画どおりでも5年で黒字にならない（5年目▲${mil(-Y5.op)}百万円）。効果が出た班から15台に増やせれば5年目に黒字（${mil(D15.y5.op)}百万円）、新規が計画の6割でも黒字`, { breakLine: true }),
      run("そのため、2年目末に「平均12台」を判断基準に置く", { bold: true })], 9);
    note(s, "固定費3か月分＝4年目の販管費÷4。新規拠点・個社設定を同じ割合で減らし、社員数＝4＋（計画−4）×達成度で採用も遅らせた簡易計算。15台はAIグラス12台＋表示付き3台");
  }

  // ---------- 14 ④10年で見ると ----------
  {
    const s = add(14);
    heading(s, `④ 10年の見通し：10台でも${R10.firstBlack}年目に黒字、${R10.cumBack}年目に累積を回収`, "5年目までの計画は変えずに、6〜10年目を同じ式で延長。条件は6年目から採用のペースを落とすこと");
    const top = 1.12, gw = 5.3, yrs10 = R10.rows.map((r) => r.year + "年");
    txt(s, [run("━ 売上", { bold: true }), run("　■ 営業損益", { color: ACC, bold: true }), run("　┅ 累積の営業損益（百万円）", { color: K.gray700 })], { x: X0, y: top, w: gw, h: 0.24, size: 8.5, valign: "middle" });
    s.addChart([
      { type: pres.charts.LINE, data: [{ name: "売上", labels: yrs10, values: R10.rows.map((r) => mil(r.sales)) }], options: { chartColors: [K.ink], lineSize: 3, lineDataSymbol: "circle", lineDataSymbolSize: 5, showValue: false } },
      { type: pres.charts.LINE, data: [{ name: "営業損益", labels: yrs10, values: R10.rows.map((r) => mil(r.op)) }], options: { chartColors: [ACC], lineSize: 2.5, lineDataSymbol: "square", lineDataSymbolSize: 5, showValue: true } },
      { type: pres.charts.LINE, data: [{ name: "累積", labels: yrs10, values: R10.rows.map((r) => mil(r.cum)) }], options: { chartColors: [K.gray500], lineSize: 2, lineDash: "dash", lineDataSymbol: "none", showValue: false } },
    ], { x: X0, y: top + 0.26, w: gw, h: 2.5, catAxisLabelColor: K.gray700, valAxisLabelColor: K.gray700, catAxisLabelFontSize: 8.5, valAxisLabelFontSize: 8,
      catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", valGridLine: { color: K.rule, size: 0.5 }, catGridLine: { style: "none" },
      valAxisMinVal: -200, valAxisMaxVal: 700, valAxisMajorUnit: 100, valAxisLineShow: false, showLegend: false, catAxisLabelPos: "low",
      dataLabelFontSize: 7.5, dataLabelFontFace: "+mn-lt", dataLabelColor: ACC, dataLabelPosition: "t" });
    table(s, [["", "5年目", "6年目", "8年目", "10年目"], ["拠点", ...[4, 5, 7, 9].map((i) => String(R10.rows[i].sites))], ["社員", ...[4, 5, 7, 9].map((i) => String(R10.rows[i].staff))], ["売上（百万円）", ...[4, 5, 7, 9].map((i) => String(mil(R10.rows[i].sales)))]],
      { x: X0, y: top + 2.85, w: gw, colW: [1.5, 0.95, 0.95, 0.95, 0.95], rowH: 0.24, size: 8.5, name: "kpi10" });
    const rx = 5.95, rw = 3.65;
    txt(s, "前提が崩れても10年で黒字になるか", { x: rx, y: top, w: rw, h: 0.26, size: 11, bold: true });
    const TEN = [["計画どおり（10台）", {}], ["1拠点15台", { plan: D15plan }], ["解約 年20%", { churnRate: 0.2 }], ["値下げ −20%", { priceK: 0.8 }], ["AI・保存費が2倍", { perDevice: V.cloudPerDevice * 2 }]]
      .map(([n, o]) => { const r = V.run({ years: 10, ...o }); return [n, r.firstBlack ? r.firstBlack + "年目" : "─", { text: r.cumBack ? r.cumBack + "年目" : "─", color: r.cumBack ? K.ink : ACC, bold: !r.cumBack }, { text: String(mil(r.minCash)), color: r.minCash < 0 ? ACC : K.ink, bold: r.minCash < 0 }]; });
    table(s, [["10年で", "黒字", "赤字を回収", "最低現金"], ...TEN], { x: rx, y: top + 0.3, w: rw, colW: [1.35, 0.7, 0.85, 0.75], rowH: 0.28, size: 8.5, align: "center", leftCols: [0], strongRows: [1], name: "stress10" });
    darkBox(s, rx, top + 2.12, rw, 1.66, [lab("ポイント", true), run("", { breakLine: true }),
      run(`利用料は毎年積み上がるため、6年目以降は利益が伸びる。10年目の営業利益は約${mil(R10.rows[9].op)}百万円（15台なら約${mil(R10_15.rows[9].op)}百万円）`, { breakLine: true }),
      run("値下げやAI費用2倍の場合は、現金が一時的にマイナスになり、自己資金の追加が必要", { color: DARK })], 9);
    note(s, `6〜10年目の前提：新規 年36→40拠点（関東の中堅で頭打ち）、社員24→32人、解約は年10%。電線工場・海外は入れていない。10年目でターゲット市場の約${(R10.rows[9].sites * 300 / 141e2).toFixed(1)}%`);
  }

  // ---------- 15 ⑤段階ごとの課題・実現可能性 ----------
  {
    const s = add(15);
    heading(s, "⑤ 段階ごとの課題と、計画の弱いところ", "各段階の一番の課題と打ち手。計画を見直して見つけた弱点は、確かめ方とセットで示す");
    const st = [
      ["準備期", "0〜6か月", "効果が出るか", "グラスの電池・録画の制約で記録が途切れる → 要所だけ記録、写真台帳から自動に"],
      ["導入期", "1〜2年目", "売れるか・続くか", "300万円を払うか未検証。監視と感じて使わない → 有償試験導入、記録の範囲と労使合意を先に"],
      ["成長期", "3〜5年目", "台数と人", "10台では黒字にならない。採用が追いつかない → 台数を増やす提案、採用を拠点数に連動"],
      ["成熟期", "6年目〜", "大手との競争", "日立などのAI化。データの持ち主 → 工種の型と監修で差、データ契約を最初から、提携"],
    ];
    const cw = (W - 0.3) / 4, top = 1.12;
    st.forEach(([ph, when, key, d], i) => {
      const x = X0 + i * (cw + 0.1);
      box(s, { x, y: top, w: cw, h: 0.55, fill: i === 0 ? ACC : i === 3 ? K.ink : K.gray700 });
      txt(s, [run(ph, { bold: true, fontSize: 11.5 }), run("　" + when, { fontSize: 8.5, breakLine: true }), run("課題：" + key, { bold: true, fontSize: 9.5 })], { x: x + 0.08, y: top, w: cw - 0.16, h: 0.55, color: K.paper, valign: "middle" });
      box(s, { x, y: top + 0.6, w: cw, h: 1.0, fill: K.fill });
      txt(s, d, { x: x + 0.08, y: top + 0.63, w: cw - 0.16, h: 0.94, size: 8.5, valign: "top" });
    });
    table(s, [["弱いところ", "何が問題か", "どう確かめる・直す"],
      ["普段使いのグラス", "動画1回3分程度まで（報道）・720p・連続撮影で電池が持たない・保護めがねの規格", "要所の短い記録に絞る。準備期に実機を1日使う"],
      ["値段", `年300万円は未検証。得は約${CV.ratio.toFixed(1)}倍で余裕が小さい`, "試験導入3社に価格を聞く"],
      ["黒字化", "10台のままでは5年で黒字にならない", "2年目末に平均12台を線にする"],
      ["人とCAC", "CTO・コンサル8人を採れるか。CACに創業者の営業時間がない", "採用を拠点数に連動。実績で更新"],
      ["最大のリスクは自分", "創業者が律速になる・前職との関係", "役割を絞り判断基準を先に。前職の顧客と機密は使わない"]],
      { x: X0, y: 2.85, w: W, colW: [1.6, 4.3, 3.3], rowH: [0.26, 0.38, 0.32, 0.32, 0.32, 0.32], size: 8.5, leftCols: [1, 2], name: "weak" });
    note(s, "ほかに：データがメーカーのアプリを経由する（規約と設定を確認）、解約率10%に日本の根拠がない（更新時の継続率で差し替える）");
  }

  // ---------- 16 ⑤準備期・最後に ----------
  {
    const s = add(16);
    heading(s, "⑤ 準備期の6か月と、最後に", "最初に確かめるのは「効果が出るか」。利益は目的ではなく、事業を続けるための条件");
    const top = 1.12, lw = 4.3;
    txt(s, "準備期の費用（約500万円・自己資金）", { x: X0, y: top, w: lw, h: 0.26, size: 11, bold: true });
    const uses = [["試作品の開発", 250], ["1拠点での実証", 100], ["グラス・機材", 80], ["設立・契約・知財", 40], ["予備", 30]];
    const k = 2.0 / 250;
    uses.forEach(([n, v], i) => {
      const y = top + 0.3 + i * 0.27;
      txt(s, n, { x: X0, y, w: 1.55, h: 0.25, size: 9, valign: "middle" });
      box(s, { x: X0 + 1.6, y: y + 0.05, w: v * k, h: 0.16, fill: i === 0 ? ACC : K.ink });
      txt(s, String(v), { x: X0 + 1.65 + v * k, y, w: 0.5, h: 0.25, size: 9, bold: true, valign: "middle" });
    });
    [["3か月", "試作品：記録 → 写真の自動抽出 → 報告書"], ["6か月", `1拠点で実証：1台1日${be}分減ったか、AIグラスで1日記録できるか`], ["6か月", "有償試験導入3拠点の契約。合格したら2億円を入れて採用を始める"]].forEach(([t, v], i) => {
      const y = top + 1.72 + i * 0.4;
      box(s, { x: X0, y: y + 0.04, w: 0.62, h: 0.3, fill: K.ink });
      txt(s, t, { x: X0, y: y + 0.04, w: 0.62, h: 0.3, size: 9, bold: true, color: K.paper, align: "center", valign: "middle" });
      txt(s, v, { x: X0 + 0.7, y, w: lw - 0.7, h: 0.38, size: 9, valign: "middle" });
    });
    const rx = 4.95, rw = 4.65;
    box(s, { x: rx, y: top, w: rw, h: 1.0, fill: K.fill });
    txt(s, [run("10年で現場に戻る時間", { fontSize: 9.5, breakLine: true }), run(`約${man(Math.round(IMPACT.hours / 1e4))}万時間`, { fontSize: 20, bold: true, color: ACC }), run(`　約${Math.round(IMPACT.value / 1e4)}億円分の書類の時間`, { fontSize: 9 })],
      { x: rx + 0.14, y: top, w: rw - 0.28, h: 1.0, valign: "middle" });
    txt(s, "事業への向き合い方（社会起業家として）", { x: rx, y: top + 1.08, w: rw, h: 0.26, size: 11, bold: true });
    [["利益は、事業を続けるための条件", "配当は求めず、現場支援と開発に回す"], ["自己資金は見返りを求めずに出す", "共感してくれる工事会社・職人・業界団体と一緒に育てる"], ["教材の一部は業界に無償で公開する", "業界団体や学校で使えるようにする"], ["撤退ラインは守る", "資金には限りがある。続けられる形を探し続ける"]].forEach(([h, d], i) => {
      const y = top + 1.38 + i * 0.4;
      box(s, { x: rx, y: y + 0.05, w: 0.3, h: 0.3, fill: i === 3 ? K.gray700 : ACC });
      txt(s, String(i + 1), { x: rx, y: y + 0.05, w: 0.3, h: 0.3, size: 10, bold: true, color: K.paper, align: "center", valign: "middle" });
      txt(s, [run(h, { bold: true }), run("　" + d, { fontSize: 8.5, color: K.gray700 })], { x: rx + 0.38, y, w: rw - 0.38, h: 0.4, size: 9.5, valign: "middle" });
    });
    box(s, { x: X0, y: 4.2, w: W, h: 0.68, fill: K.ink });
    txt(s, [run("日本の現場の勘を、人に依存しない形で次の担い手へ残したい。", { bold: true, fontSize: 12.5, breakLine: true }), run("最初の目標は、6か月で1拠点を立ち上げることです。", { fontSize: 10 })],
      { x: X0 + 0.18, y: 4.2, w: W - 0.36, h: 0.68, color: K.paper, valign: "middle", psa: 2 });
    note(s, "時間と価値は10年計画（10台）の延べ拠点数×1拠点の削減（年1,400時間・約500万円）。削減時間（1台1日35分）は仮説で、実証で測る");
  }

  fs.mkdirSync(OUT_DIR, { recursive: true });
  await pres.writeFile({ fileName: OUT });
  console.log("wrote", OUT, "台本の合計", NOTES.total);
}

main().catch((e) => { console.error(e); process.exit(1); });
