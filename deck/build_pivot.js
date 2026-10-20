// カンツギ 事業計画書「10分・16枚版」: ビジネス映像メディア風（大きな見出し・1枚1メッセージ・大きな数字・黒白＋黄色のマーカー）。
// A4の表裏に8枚ずつ印刷しても読めるよう、本文は14pt以上。数字はすべて tora_v2.js から。
// Usage: node build_pivot.js [outDir]   (needs pptxgenjs and sharp)
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const L = require("./tora_lib.js");
const V = require("./tora_v2.js");

const OUT_DIR = process.argv[2] || __dirname;
const OUT = path.join(OUT_DIR, "カンツギ_事業計画書_10分16枚版.pptx");
const F = L.F;
const INK = "111111", PAPER = "FFFFFF", YEL = "FFE14D", ACC = "B8692E", GRAY = "5E5E5E", LIGHT = "F1F1F1", RULE = "CFCFCF";

// ---------- numbers ----------
const CV = V.customerValue(), U = V.unit(), R = V.run().rows, Y5 = R[4];
const R10 = V.run({ years: 10 }), D15 = V.run({ plan: { devices: 15, mix: { glass: 12, display: 3 } } });
const be = Math.round(CV.breakEvenMin);
const man = (v) => Math.round(v).toLocaleString("en-US");
const mil = (v) => Math.round(v / 100);
const fund = V.GROWTH.funding.reduce((a, b) => a + b, 0) + 500;
const siteYears = R10.rows.reduce((a, r) => a + r.sites, 0);
const IMPACT = { hours: siteYears * CV.freedHours, value: siteYears * CV.docs };

// ---------- primitives ----------
const t = (s, text, o) => s.addText(text, {
  fontFace: F, fontSize: o.size || 16, color: o.color || INK, bold: o.bold !== false && !o.light, x: o.x, y: o.y, w: o.w, h: o.h,
  margin: o.margin ?? 0, valign: o.valign || "middle", align: o.align || "left", lineSpacingMultiple: o.lsm || 1.1,
  paraSpaceAfter: o.psa || 0, fill: o.fill ? { color: o.fill } : undefined, isTextBox: true,
});
const r = (text, o = {}) => ({ text, options: { fontFace: F, ...o } });
const mark = (text, o = {}) => r(text, { highlight: YEL, ...o }); // 黄色のマーカー
const rect = (s, x, y, w, h, fill, line) => s.addShape("rect", { x, y, w, h, fill: { color: fill }, line: line ? { color: line, width: 1.5 } : { type: "none" } });

const SECTION = { 1: "", 2: "① 事業", 3: "① 事業", 4: "① 理由と想い", 5: "① 事業", 6: "① 競合", 7: "② スタッフ", 8: "③ 経営理念",
  9: "④ 経営計画", 10: "④ 経営計画", 11: "④ 経営計画", 12: "④ 経営計画", 13: "④ 経営計画", 14: "⑤ 課題", 15: "⑤ 課題", 16: "最後に" };
// 見出し：左上に黒い章ラベル、その下に大きな一文（マーカー付き）
function head(s, n, parts, sub) {
  t(s, SECTION[n], { x: 0.4, y: 0.28, w: 1.9, h: 0.36, size: 13, color: PAPER, fill: INK, align: "center" });
  t(s, parts, { x: 0.4, y: 0.72, w: 9.2, h: 0.82, size: 29, lsm: 1.0 });
  if (sub) t(s, sub, { x: 0.4, y: 1.5, w: 9.2, h: 0.36, size: 15, color: GRAY, bold: false, light: true });
}
// 下の帯（テロップ）
function telop(s, text) {
  rect(s, 0, 4.78, 10, 0.5, INK);
  t(s, text, { x: 0.4, y: 4.78, w: 8.0, h: 0.5, size: 16, color: PAPER });
}
// 大きな数字のカード
function stat(s, x, y, w, h, num, label, o = {}) {
  rect(s, x, y, w, h, o.fill || LIGHT);
  t(s, num, { x: x + 0.15, y: y + 0.08, w: w - 0.3, h: h * 0.58, size: o.size || 40, color: o.numColor || INK, align: o.align || "left" });
  t(s, label, { x: x + 0.15, y: y + h * 0.6, w: w - 0.3, h: h * 0.36, size: o.labelSize || 14, color: o.labelColor || GRAY, bold: false, light: true, valign: "top", align: o.align || "left" });
}

// ---------- 台本（ポッドキャストのように、話し言葉で。聞き取りやすい速さ＝1分約300字で時間を出す） ----------
const NOTE_LIST = [null,
  ["つかみ", `こんにちは。今日は、私が立ち上げる会社「カンツギ」の話をさせてください。カンツギは、勘を継ぐ、と書きます。割れたお茶碗を金でつなぐ「金継ぎ」ってありますよね。あれと同じで、途切れそうな現場の勘を、データでつなぐ会社です。10分で、何をやるのか、誰とやるのか、お金はどうなるのか、全部お話しします。`],
  ["一言で", `まず、一言で言うとこうです。メガネを掛けて、いつも通り働くだけ。それで日報と写真台帳ができあがる。使うのは、普段のメガネと同じくらい軽い、カメラ付きのAIグラスです。記録して、書類にして、最後は若手の教材にする。この三つを一続きでやります。お客さんは、関東の中堅の電気工事会社です。`],
  ["なぜ今", `じゃあ、なぜ今なのか。数字を三つだけ見てください。建設業で働く人の36.7%が、55歳以上です。2024年からは、建設業にも残業の上限ができました。原則、月45時間です。そして、やり直しの工事、いわゆる手戻りは、建設全体で年1.5兆円と見積もれます。人は減る、時間は増やせない、ミスは高くつく。だから、熟練の人の勘を残す仕組みが、今いるんです。`],
  ["理由と想い", `ここで少し、私自身の話をさせてください。私は電線メーカーで、地中送電線の施工管理を5年やってきました。そこで何度も見たのが、ベテランの「ここが大事なんだよ」という判断が、手順書に書かれないまま、退職と一緒に消えていく場面です。現場のこともAIのことも分かる人は、まだ多くありません。だから、私がやります。`],
  ["使い方", `使い方はシンプルです。メガネは全員には配りません。1つの班に1台、職長さんが掛けます。1つの支店で10台です。撮るのは作業エリアだけ。休憩中や移動中は撮りません。録画はメガネの中に残して、作業の区切りや「ここ撮って」という声の場面だけをAIに送ります。そこから写真台帳と日報ができて、ベテランの判断は教材として残ります。`],
  ["競合", `正直に言います。ほぼ同じ方向のサービスを、日立グループがもうやっています。ビルの保守で安全を見守るAI Safety、それから日立パワーソリューションズの、変電設備の点検でミスを見つけるAIです。技術と規模では、正直かないません。だから、ずらします。日立は大手の保守と安全。私たちは、中堅の電気工事会社の、工事そのものの書類と、技能の継承です。月額で、2か月で使い始められる形にします。`],
  ["スタッフ", `一緒に働くメンバーです。5年目に23人の会社にします。カギは三つの役割です。映像AIを作るエンジニア。電気工事の経験が10年以上あって、映像のどこが大事かを見分ける現場監修。そして、お客さんの現場に入って、その会社の書類の形に合わせて設定する導入コンサルタント。この設定は外注しません。社内でやるから、ノウハウが会社に貯まるんです。私の役割は、最初の営業と、判断の基準をつくることです。`],
  ["経営理念", `経営理念は、「現場の勘を次の担い手へ」です。ホームページでも、最初にこの言葉を出します。大事にしたいことは三つ。現場を主役にすること。入力の手間は増やしません。勘に敬意を払うこと。記録は本人の同意のもとで、誰の技術かを残します。そして、数字で約束すること。効果は毎月測って見せます。特に現場の方には、これは監視じゃない、あなたの技術を残す仕組みだ、と伝えたいです。`],
  ["お客さんの得", `では、お客さんはいくら得をするのか。メガネを掛けた班で、職長さんの日報が1日15分、施工管理者の写真整理が20分減るとします。10台で計算すると、年に約${man(Math.round(CV.docs / 10) * 10)}万円分の時間です。払っていただくのは、利用料とメガネのリースで約${man(CV.cost)}万円。だいたい${CV.ratio.toFixed(1)}倍ですね。逆に言うと、1台1日${be}分減れば元が取れます。ここを、最初の実証の合格ラインにします。`],
  ["稼ぎ方", `私たちの稼ぎ方です。会社の基盤に月5万円、メガネ1台に月2万円。10台で月25万円、年300万円です。報告書の形に整えるところまで、この中に入っています。ここからAIと保存の費用を引くと、1拠点あたり年${man(U.gpYear)}万円が残ります。粗利率は${Math.round(V.marginSite() * 100)}%。スマホの家族プランと同じで、去年のお客さんが今年も払ってくれるので、売上は雪だるま式に積み上がります。`],
  ["5年の計画", `5年の計画です。1年目は3社で試してもらうところから。5年目には${Y5.sites}拠点、売上は約${(Y5.sales / 1e4).toFixed(1)}億円です。ただ、ここは正直に言います。人を先に雇うので、5年間はずっと赤字です。5年目もまだ約${mil(-Y5.op)}百万円の赤字。これを隠さずにお見せしたうえで、じゃあ10年ならどうか、を次で見てください。`],
  ["10年", `同じ計算を10年まで延ばしました。5年目までの計画は変えずに、6年目からは採用のペースを落とします。すると、1拠点10台のままでも、${R10.firstBlack}年目に黒字になって、${R10.cumBack}年目には5年分の赤字を取り戻します。10年目の利益は約${mil(R10.rows[9].op)}百万円。もし、効果が出た班からメガネを15台に増やしてもらえたら、5年目で黒字です。時間を味方にできる事業だ、というのがポイントです。`],
  ["資金", `お金は、自分の資金でまかないます。合計で約${(fund / 1e4).toFixed(2)}億円です。ただ、一度に全部は入れません。最初は500万円で試作品と実証。実証で合格したら、1年目に2億円。3年目に、お客さんが続けてくれていたら、5,000万円を追加します。そして、止まる線も先に決めておきます。実証で1台1日15分も減らなかったら、撤退です。`],
  ["段階ごとの課題", `それぞれの段階の課題です。準備期は、本当に効果が出るか。普段使いのメガネは、録画の長さや電池に限りがあります。導入期は、売れるか、使い続けてもらえるか。成長期は、メガネの台数と、人を採れるか。そして成熟期は、日立のような大手との競争です。どの段階にも、進む条件と止まる条件を置いています。`],
  ["弱いところ", `最後に、自分で計画を疑って見つけた弱いところです。一つ目、普段使いのメガネは、現場にはまだ弱い。落としたら壊れるし、保護めがねの規格も確認が必要です。二つ目、年300万円という値段は、まだ誰にも確かめていません。三つ目、10台のままだと、5年では黒字になりません。四つ目、日立が中堅向けに来たら、ブランドでは負けます。だからこそ、最初の6か月で、ここを一つずつ確かめます。`],
  ["締め", `最後に、私の願いです。10年続けば、書類に使っていた約${man(Math.round(IMPACT.hours / 1e4))}万時間が、現場と若手の指導に戻ります。私にとって、利益は目的ではなく、続けるための条件です。配当は求めません。共感してくれる会社や職人さんと一緒に、育てていきたい。日本の現場の勘を次の担い手へ。その最初の一歩として、6か月で、最初の1拠点を動かします。ご清聴ありがとうございました。`],
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
  pres.layout = "LAYOUT_16x9";
  pres.title = "カンツギ 事業計画書（10分・16枚版）";
  pres.theme = { headFontFace: F, bodyFontFace: F };
  const logo = await L.png("kantsugi-logotype-color");
  const logoW = await L.png("kantsugi-logotype-reverse");
  pres.defineSlideMaster({ title: "P", background: { color: PAPER },
    objects: [{ image: { x: 8.45, y: 0.3, w: 1.15, h: 1.15 * 120 / 560, data: logo } }],
    slideNumber: { x: 9.2, y: 5.3, w: 0.4, h: 0.25, fontFace: F, fontSize: 10, color: GRAY, align: "right" } });
  pres.defineSlideMaster({ title: "C", background: { color: INK } });
  const add = (n, m = "P") => { const s = pres.addSlide({ masterName: m }); s.addNotes(NOTES[n]); return s; };

  // 1 表紙
  {
    const s = add(1, "C");
    s.addImage({ data: logoW, x: 0.6, y: 0.5, w: 2.2, h: 2.2 * 120 / 560 });
    t(s, [r("現場の勘を", { breakLine: true }), r("次の担い手へ", { highlight: YEL, color: INK })], { x: 0.6, y: 1.25, w: 8.8, h: 2.0, size: 54, color: PAPER, lsm: 1.15 });
    t(s, "AIグラス × 電気工事。メガネを掛けて働くだけで、書類と教材ができる", { x: 0.6, y: 3.4, w: 8.8, h: 0.5, size: 18, color: PAPER, bold: false, light: true });
    rect(s, 0.6, 4.2, 4.6, 0.6, ACC);
    t(s, "株式会社カンツギ　事業計画（10分）", { x: 0.75, y: 4.2, w: 4.4, h: 0.6, size: 16, color: PAPER });
  }

  // 2 一言で
  {
    const s = add(2);
    head(s, 2, [r("メガネを掛けるだけで、"), mark("日報と台帳ができる")], "普段のメガネと同じ軽さのAIグラスで、熟練者の作業を記録する");
    [["記録", "AIグラスが\n要所を自動で撮る"], ["書類", "写真台帳・日報を\n各社の形で作る"], ["教材", "ベテランの判断を\n若手に渡す"]].forEach(([h, d], i) => {
      const x = 0.4 + i * 3.15;
      rect(s, x, 2.1, 2.85, 2.35, i === 2 ? INK : LIGHT);
      t(s, `${i + 1}`, { x: x + 0.2, y: 2.2, w: 0.6, h: 0.6, size: 30, color: i === 2 ? YEL : ACC });
      t(s, h, { x: x + 0.2, y: 2.75, w: 2.5, h: 0.65, size: 30, color: i === 2 ? PAPER : INK });
      t(s, d, { x: x + 0.2, y: 3.45, w: 2.5, h: 0.85, size: 15, color: i === 2 ? PAPER : GRAY, bold: false, light: true, valign: "top" });
      if (i < 2) t(s, "▶", { x: x + 2.87, y: 3.0, w: 0.28, h: 0.5, size: 18, color: ACC, align: "center" });
    });
    telop(s, "お客様：関東の中堅の電気工事会社　／　1拠点10台で年300万円");
  }

  // 3 なぜ今
  {
    const s = add(3);
    head(s, 3, [r("人は減る。時間は増やせない。"), mark("ミスは高くつく")], "だから今、熟練者の勘を残す仕組みが要る");
    [["36.7%", "建設業で働く人のうち\n55歳以上（2024年）"], ["月45時間", "2024年から建設業にも\n残業の上限（原則）"], ["年1.5兆円", "建設全体の手戻り\n（建設投資×2%で推計）"]].forEach(([n, l], i) => {
      stat(s, 0.4 + i * 3.1, 2.05, 2.9, 2.45, n, l, { size: 34, numColor: i === 2 ? ACC : INK, fill: LIGHT });
    });
    t(s, "出典：国土交通省（労働力調査 2024年）、労働基準法、国交省 建設投資見通し×海外研究の手戻り率", { x: 0.4, y: 4.55, w: 9.2, h: 0.2, size: 9, color: GRAY, bold: false, light: true });
    telop(s, "熟練者が辞めると、判断の理由も一緒に消える");
  }

  // 4 理由と想い
  {
    const s = add(4);
    head(s, 4, [r("ベテランの勘が"), mark("消えていくのを見てきた")], "選んだ理由は3つ。払う理由がある市場、使える技術、そして自分");
    [["市場", "人手不足と残業の上限。お金を払ってでも解きたい"], ["技術", "見て理解するAIと、50gのAIグラスが実用に"], ["自分", "地中送電線の施工管理5年。AIも自分で作れる"]].forEach(([h, d], i) => {
      const y = 2.0 + i * 0.62;
      t(s, h, { x: 0.4, y, w: 1.1, h: 0.52, size: 18, color: PAPER, fill: i === 2 ? ACC : INK, align: "center" });
      t(s, d, { x: 1.65, y, w: 4.2, h: 0.52, size: 15, bold: false, light: true });
    });
    rect(s, 6.1, 2.0, 3.5, 2.5, INK);
    t(s, [r("想い", { color: YEL, fontSize: 16, breakLine: true }), r("割れた器を金で継ぐ「金継ぎ」のように、途切れそうな現場の勘を、データで継ぐ", { fontSize: 18 })], { x: 6.3, y: 2.1, w: 3.1, h: 2.3, color: PAPER, psa: 6 });
    telop(s, "社名「カンツギ（勘継）」は、この想いから");
  }

  // 5 使い方
  {
    const s = add(5);
    head(s, 5, [r("1班に1台、"), mark("職長が掛ける"), r("だけ")], "全員には配らない。撮るのは作業エリアの、大事な場面だけ");
    const rows = [["誰が", "1班に1台、職長が掛ける"], ["何台", "1支店10台（5現場×2班）"], ["どこで", "作業エリアだけ。休憩・移動は撮らない"], ["何を", "作業の区切り・「ここ撮って」の声の場面"]];
    rows.forEach(([k, v], i) => {
      const y = 2.0 + i * 0.62;
      t(s, k, { x: 0.4, y, w: 1.2, h: 0.52, size: 17, color: PAPER, fill: INK, align: "center" });
      t(s, v, { x: 1.75, y, w: 4.3, h: 0.52, size: 15, bold: false, light: true });
    });
    stat(s, 6.3, 2.0, 3.3, 1.15, "約50g", "カメラ付きAIグラス（普段のメガネの重さ）", { size: 32, numColor: ACC, labelSize: 12 });
    stat(s, 6.3, 3.3, 3.3, 1.15, "8台＋2台", "記録用8台＋図面を映す表示付き2台", { size: 30, labelSize: 12 });
    telop(s, "録画はメガネに残し、AIに送るのは選んだ場面だけ");
  }

  // 6 競合・差別化
  {
    const s = add(6);
    head(s, 6, [r("日立と同じ方向。"), mark("だから、ずらす")], "日立グループ（AI Safety・日立パワーソリューションズ）は、大手の保守と安全から");
    const rows = [["", "日立グループ", "カンツギ"], ["仕事", "設備の保守・点検", "電気工事の施工"], ["お客様", "自社・大手", "中堅の電気工事会社"], ["価値", "安全・ミスの照合", "書類の削減と技能継承"], ["買い方", "個別の開発・保守契約", "月額・2か月で導入"]];
    rows.forEach((row, i) => {
      const y = 1.98 + i * 0.5;
      row.forEach((c, j) => {
        const x = [0.4, 1.85, 5.75][j], w = [1.35, 3.8, 3.85][j];
        if (i === 0) t(s, c, { x, y, w, h: 0.44, size: 15, color: PAPER, fill: j === 2 ? ACC : j === 1 ? GRAY : PAPER, align: "center" });
        else t(s, c, { x, y, w, h: 0.44, size: 15, color: j === 2 ? INK : GRAY, bold: j !== 1, light: j === 1, fill: j === 0 ? LIGHT : undefined, align: j === 0 ? "center" : "left", margin: j ? 0.1 : 0 });
      });
      if (i) s.addShape("line", { x: 0.4, y: y + 0.47, w: 9.2, h: 0, line: { color: RULE, width: 0.75 } });
    });
    telop(s, "技術と規模では勝てない。勝負は「施工の型」を先に揃えること");
  }

  // 7 スタッフ
  {
    const s = add(7);
    head(s, 7, [r("5年目"), mark(`${Y5.staff}人`), r("。カギは3つの役割")], "帳票の設定は外注せず、社内のコンサル部隊で。ノウハウを会社に貯める");
    [["エンジニア", "1→5人", "映像AIとデータ基盤をつくる"], ["現場監修", "1→2人", "電気工事10年以上。映像の\n大事な場面を見分ける"], ["導入コンサル", "1→8人", "現場に入り、各社の書類の\n形に合わせて設定する"]].forEach(([h, n, d], i) => {
      const x = 0.4 + i * 3.1;
      rect(s, x, 2.0, 2.9, 2.0, LIGHT);
      t(s, h, { x: x + 0.15, y: 2.08, w: 2.6, h: 0.45, size: 19 });
      t(s, n, { x: x + 0.15, y: 2.52, w: 2.6, h: 0.55, size: 28, color: ACC });
      t(s, d, { x: x + 0.15, y: 3.1, w: 2.6, h: 0.85, size: 13.5, color: GRAY, bold: false, light: true, valign: "top" });
    });
    t(s, [r("私の役割：", { color: ACC }), r("最初の営業と、判断の基準をつくること")], { x: 0.4, y: 4.12, w: 9.2, h: 0.5, size: 17 });
    telop(s, "ほかにCTO・営業・管理・戦略企画。社外の技術顧問と業界の顧問");
  }

  // 8 経営理念
  {
    const s = add(8, "C");
    t(s, SECTION[8], { x: 0.4, y: 0.28, w: 1.9, h: 0.36, size: 13, color: INK, fill: YEL, align: "center" });
    t(s, [r("現場の勘を", { breakLine: true }), r("次の担い手へ", { color: YEL })], { x: 0.4, y: 0.85, w: 9.2, h: 1.6, size: 46, color: PAPER, lsm: 1.1 });
    [["現場を主役に", "入力の手間を増やさない"], ["勘に敬意を", "本人の同意で、誰の技術かを残す"], ["数字で約束", "効果を毎月測って見せる"]].forEach(([h, d], i) => {
      const x = 0.4 + i * 3.1;
      s.addShape("line", { x, y: 2.85, w: 2.9, h: 0, line: { color: YEL, width: 2.5 } });
      t(s, h, { x, y: 2.95, w: 2.9, h: 0.55, size: 21, color: PAPER });
      t(s, d, { x, y: 3.5, w: 2.9, h: 0.6, size: 14.5, color: "D9D9D9", bold: false, light: true, valign: "top" });
    });
    t(s, "現場で働く方へ：これは監視ではなく、あなたの技術を残す仕組みです", { x: 0.4, y: 4.55, w: 9.2, h: 0.5, size: 16, color: YEL });
  }

  // 9 お客様の得
  {
    const s = add(9);
    head(s, 9, [r("払う額の"), mark(`約${CV.ratio.toFixed(1)}倍`), r("が、時間で戻る")], "1拠点10台。職長の日報15分＋施工管理者の写真整理20分が、1台1日で減る想定");
    stat(s, 0.4, 2.0, 2.9, 2.45, `${man(Math.round(CV.docs / 10) * 10)}万円`, "お客様の得（年）\n書類の時間だけで計算", { size: 36, numColor: ACC });
    t(s, "÷", { x: 3.3, y: 2.7, w: 0.4, h: 0.8, size: 30, align: "center" });
    stat(s, 3.7, 2.0, 2.6, 2.45, `${man(CV.cost)}万円`, "お客様が払う額（年）\n利用料＋グラスのリース", { size: 36 });
    rect(s, 6.5, 2.0, 3.1, 2.45, INK);
    t(s, [r("元が取れるのは", { fontSize: 15, breakLine: true }), r(`1台1日${be}分`, { fontSize: 28, color: YEL, breakLine: true }), r("実証の合格ライン", { fontSize: 14 })], { x: 6.65, y: 2.05, w: 2.85, h: 2.35, color: PAPER, align: "center", psa: 4 });
    telop(s, "手戻りが減れば、さらに上乗せ（実証で測ってから足す）");
  }

  // 10 稼ぎ方
  {
    const s = add(10);
    head(s, 10, [r("月25万円の利用料が、"), mark("毎年積み上がる")], "会社の基盤 月5万円 ＋ メガネ1台 月2万円 × 10台。初年度だけ導入支援200万円");
    stat(s, 0.4, 2.0, 2.9, 2.45, "年300万円", "1拠点（10台）の利用料\n報告書の形に整えるまで込み", { size: 34 });
    t(s, "−", { x: 3.3, y: 2.7, w: 0.4, h: 0.8, size: 30, align: "center" });
    stat(s, 3.7, 2.0, 2.6, 2.45, `${man(V.cogsSite())}万円`, "AI・保存の費用\nメガネ1台 月約6,000円", { size: 34 });
    rect(s, 6.5, 2.0, 3.1, 2.45, INK);
    t(s, [r("当社に残る", { fontSize: 15, breakLine: true }), r(`${man(U.gpYear)}万円`, { fontSize: 38, color: YEL, breakLine: true }), r(`粗利率 ${Math.round(V.marginSite() * 100)}%`, { fontSize: 15 })], { x: 6.65, y: 2.05, w: 2.85, h: 2.35, color: PAPER, align: "center", psa: 4 });
    telop(s, `1拠点を獲得する費用 約${man(U.cac)}万円に対し、7年で約${man(U.ltv(7))}万円が残る`);
  }

  // 11 5年の計画
  {
    const s = add(11);
    head(s, 11, [r("5年は赤字。"), mark("それでも隠さず見せる")], "人を先に雇うので、5年目までは赤字が続く（百万円）");
    const cols = ["", ...R.map((x) => x.year + "年目")];
    const rows = [cols, ["拠点", ...R.map((x) => String(x.sites))], ["社員", ...R.map((x) => String(x.staff))], ["売上", ...R.map((x) => String(mil(x.sales)))], ["営業損益", ...R.map((x) => "▲" + Math.abs(mil(x.op)))]];
    rows.forEach((row, i) => {
      const y = 2.0 + i * 0.5;
      row.forEach((c, j) => {
        const x = j === 0 ? 0.4 : 1.9 + (j - 1) * 1.54, w = j === 0 ? 1.4 : 1.44;
        t(s, c, { x, y, w, h: 0.44, size: i === 0 ? 14 : 18, color: i === 0 ? PAPER : i === 4 ? ACC : INK, fill: i === 0 ? INK : j === 0 ? LIGHT : undefined, align: "center", bold: i === 0 || i === 4 || j === 0, light: !(i === 0 || i === 4 || j === 0) });
      });
    });
    telop(s, `5年目：${Y5.sites}拠点・売上約${(Y5.sales / 1e4).toFixed(1)}億円。黒字まで、あと約${Math.ceil((Y5.sga - Y5.gp) / U.gpYear)}拠点分`);
  }

  // 12 10年で見ると
  {
    const s = add(12);
    head(s, 12, [r("10年で見れば、"), mark(`${R10.firstBlack}年目に黒字`)], `5年目までは変えずに延長。6年目から採用のペースを落とすと、${R10.cumBack}年目に赤字を取り戻す`);
    const labels = R10.rows.map((x) => x.year + "年");
    s.addChart(pres.charts.BAR, [{ name: "営業損益", labels, values: R10.rows.map((x) => mil(x.op)) }], {
      x: 0.4, y: 1.95, w: 5.9, h: 2.75, barDir: "col", chartColors: R10.rows.map((x) => (x.op < 0 ? "9A9A9A" : ACC)), varyColors: true, barGapWidthPct: 40,
      catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, catAxisLabelFontFace: F, valAxisLabelFontFace: F, catAxisLabelColor: INK, valAxisLabelColor: GRAY,
      valGridLine: { color: RULE, size: 0.5 }, showValue: true, dataLabelFontSize: 11, dataLabelFontFace: F, dataLabelColor: INK, dataLabelPosition: "outEnd", showLegend: false, valAxisLineShow: false,
    });
    stat(s, 6.5, 1.95, 3.1, 1.3, `${mil(R10.rows[9].op)}百万円`, "10年目の営業利益（10台）", { size: 30, numColor: ACC, labelSize: 13 });
    stat(s, 6.5, 3.4, 3.1, 1.3, "5年目に黒字", "効果が出た班から15台に増やせたら", { size: 28, labelSize: 13 });
    telop(s, "積み上げ型なので、時間が味方になる（百万円）");
  }

  // 13 資金
  {
    const s = add(13);
    head(s, 13, [r("自己資金を、"), mark("線を越えたときだけ"), r("入れる")], `合計 約${(fund / 1e4).toFixed(2)}億円。資金が豊富でも、一度に全部は使わない`);
    [["準備期", "500万円", "試作品と実証"], ["1年目", "2億円", `実証で1台1日${be}分以上`], ["3年目", "5,000万円", "継続率90%・平均12台"]].forEach(([w, n, c], i) => {
      const x = 0.4 + i * 3.1;
      rect(s, x, 2.0, 2.9, 1.8, i === 1 ? INK : LIGHT);
      t(s, w, { x: x + 0.15, y: 2.05, w: 2.6, h: 0.45, size: 17, color: i === 1 ? YEL : ACC });
      t(s, n, { x: x + 0.15, y: 2.5, w: 2.6, h: 0.65, size: 32, color: i === 1 ? PAPER : INK });
      t(s, "条件：" + c, { x: x + 0.15, y: 3.2, w: 2.6, h: 0.5, size: 13.5, color: i === 1 ? PAPER : GRAY, bold: false, light: true });
      if (i < 2) t(s, "▶", { x: x + 2.9, y: 2.65, w: 0.2, h: 0.4, size: 14, color: ACC, align: "center" });
    });
    t(s, [r("止まる線：", { color: ACC }), r("実証で1台1日15分も減らなければ撤退。2年目末に平均12台に届かなければ採用を止める")], { x: 0.4, y: 3.95, w: 9.2, h: 0.65, size: 15 });
    telop(s, "融資・出資は使わない。返済も利息もない");
  }

  // 14 段階ごとの課題
  {
    const s = add(14);
    head(s, 14, [r("段階ごとに、"), mark("一番の課題"), r("は変わる")], "それぞれに「進む条件」と「止まる条件」を置いている");
    [["準備期", "効果が出るか", "メガネの電池と録画の制約"], ["導入期", "売れるか・続くか", "300万円を払うか、監視と思われないか"], ["成長期", "台数と人", "10台では黒字にならない・採用"], ["成熟期", "大手との競争", "日立が中堅・施工に来たら"]].forEach(([p, k, d], i) => {
      const x = 0.4 + i * 2.33;
      t(s, p, { x, y: 2.0, w: 2.2, h: 0.5, size: 18, color: PAPER, fill: i === 0 ? ACC : INK, align: "center" });
      rect(s, x, 2.55, 2.2, 1.95, LIGHT);
      t(s, k, { x: x + 0.12, y: 2.62, w: 1.96, h: 0.75, size: 19, valign: "top" });
      t(s, d, { x: x + 0.12, y: 3.4, w: 1.96, h: 1.0, size: 13.5, color: GRAY, bold: false, light: true, valign: "top" });
    });
    telop(s, "準備期の6か月で、最初の課題「効果が出るか」を確かめる");
  }

  // 15 弱いところ
  {
    const s = add(15);
    head(s, 15, [r("自分で見つけた、"), mark("計画の弱いところ")], "隠さず、確かめ方とセットで示す");
    [["普段使いのメガネは、現場に弱い", "落下・粉じん・保護めがねの規格。実機で1日使って確かめる"], ["年300万円は、まだ未検証", "試験導入の3社に値段を聞く"], ["10台のままでは、5年で黒字にならない", "効果が出た班から台数を増やす提案を仕組みに"], ["日立が来たら、ブランドで負ける", "施工の書類の型を先に揃える。提携も選ぶ"]].forEach(([h, d], i) => {
      const y = 2.0 + i * 0.68;
      t(s, String(i + 1), { x: 0.4, y, w: 0.55, h: 0.58, size: 22, color: PAPER, fill: ACC, align: "center" });
      t(s, [r(h, { fontSize: 17, breakLine: true }), r("→ " + d, { fontSize: 13.5, bold: false, color: GRAY })], { x: 1.1, y, w: 8.5, h: 0.6 });
    });
    telop(s, "弱みを先に言い、止まる線を先に引く");
  }

  // 16 最後に
  {
    const s = add(16, "C");
    t(s, SECTION[16], { x: 0.4, y: 0.28, w: 1.9, h: 0.36, size: 13, color: INK, fill: YEL, align: "center" });
    t(s, [r("儲けより、", { breakLine: true }), r("残したいものがある。", { color: YEL })], { x: 0.4, y: 0.8, w: 5.8, h: 1.6, size: 38, color: PAPER, lsm: 1.1 });
    t(s, [r("10年で現場に戻る時間", { fontSize: 15, color: "D9D9D9", breakLine: true }), r(`約${man(Math.round(IMPACT.hours / 1e4))}万時間`, { fontSize: 32, color: YEL, breakLine: true }), r(`約${Math.round(IMPACT.value / 1e4)}億円分の書類の時間`, { fontSize: 14, color: "D9D9D9" })], { x: 6.3, y: 0.8, w: 3.3, h: 1.7, color: PAPER, align: "right" });
    [["利益は目的ではなく、続けるための条件", "配当は求めず、現場と開発に戻す"], ["見返りを求めない投資", "共感する会社・職人と一緒に育てる"], ["それでも止まる線は守る", "資金は有限。続けられる形を探す"]].forEach(([h, d], i) => {
      const y = 2.65 + i * 0.6;
      s.addShape("line", { x: 0.4, y, w: 9.2, h: 0, line: { color: "444444", width: 1 } });
      t(s, h, { x: 0.4, y: y + 0.05, w: 5.4, h: 0.5, size: 17, color: PAPER });
      t(s, d, { x: 5.9, y: y + 0.05, w: 3.7, h: 0.5, size: 14, color: "D9D9D9", bold: false, light: true });
    });
    rect(s, 0, 4.6, 10, 0.68, YEL);
    t(s, "日本の現場の勘を次の担い手へ。最初の一歩として、6か月で1拠点を動かします。", { x: 0.4, y: 4.6, w: 9.2, h: 0.68, size: 17, color: INK });
  }

  fs.mkdirSync(OUT_DIR, { recursive: true });
  await pres.writeFile({ fileName: OUT });
  console.log("wrote", OUT, "台本の合計", NOTES.total);
}
main().catch((e) => { console.error(e); process.exit(1); });
