// カンツギ 事業計画書「やさしい版（マンガ）」: story and the money mechanics in comic panels.
// 見た目は16枚版と同じ（tora_lib の drawing()：図面の枠・紺・安全色。ヘルメットは黄、緑十字）。数字は tora_v2.js から。
// Usage: node build_tora_easy.js [outDir]   (needs pptxgenjs and sharp)
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const L = require("./tora_lib.js").drawing();
const { run: model, unit, shortfall } = require("./tora_model.js");
const V = require("./tora_v2.js");

const { K, S, X0, W, ACC, txt, box, hline, arrow, run, heading, panel, bubble, sfx, png, charSVG, iconSVG } = L;
const OUT_DIR = process.argv[2] || __dirname;
const OUT = path.join(OUT_DIR, "カンツギ_事業計画書_やさしい版.pptx");
const { customerValue } = require("./tora_value.js");
const CV = customerValue();
const man = (v) => Math.round(v).toLocaleString("en-US");
const PLAN = model(), R = PLAN.rows, U = unit(), Y5 = R[4], PER = Y5.sga / Y5.staff, FB = PLAN.firstBlack;
const HAVG = (15 * V.H.electrician + 20 * V.H.manager) / 35; // 1時間の人件費（職長と施工管理の平均・万円）
const M = (v) => Math.round(v / 100); // 万円 → 百万円
const be = Math.round(CV.breakEvenMin);
const F15 = V.run({ plan: { devices: 15, mix: { glass: 12, display: 3 } } }).firstBlack;
const R10 = V.run({ years: 10 }); // 10年で見ると：6年目に黒字、8年目に5年間の赤字を回収

// ---------- notes: what to say, in spoken words ----------
const NOTE_PARTS = {
  1: ["表紙・20秒", "数字は3つだけ。300万円、216万円、約4拠点",
    "カンツギです。今日はお金の仕組みを、3つの数字だけでお話しします。お客様が1拠点でメガネ10台分、年300万円払う。そのうち216万円が当社に残る。社員1人で約4拠点を見られれば黒字。この3つです。"],
  2: ["登場人物・30秒", "お金を払うのは電気工事会社。窓口は工事部長",
    "登場人物は4人です。来年定年の熟練の電工、田中さん。入社2年目の施工管理、佐藤くん。報告書と育成で手一杯の山本工事部長。そして私です。ここで大事なのは、お金を払うのは現場の人ではなく会社、つまり工事部長や社長だということです。"],
  3: ["第1話・40秒", "熟練者の勘は書かれていない。辞めたら止まる",
    "ある日の工事現場。ケーブルの接続で、田中さんは、端末の寸法はテープの巻き始めから測れと言う。佐藤くんには理由が分からない。手順書にも書いていない。工事部長は、来年田中さんが辞めたらどうなるのかと青ざめる。しかも2024年から残業に上限がかかり、工事写真と報告書で教える時間もない。建設業で働く人の3人に1人以上が55歳以上です。これが、全国の工事会社で起きていることです。"],
  4: ["第2話・40秒", "メガネを掛けて、いつも通り働くだけ",
    "そこで、普段のメガネと同じ軽さのAIグラスを掛けてもらいます。全員ではなく、1つの班に1台、職長が掛けます。田中さんは、いつも通り作業するだけ。動画と会話をAIが整理して、三つの形で渡します。若手には、図面とのズレをその場で表示。管理者には、報告書を自動で作成。そして後から、教材にします。"],
  5: ["第3話・40秒", "壁の図面が手元に出る。ミスしやすい所では止めてくれる",
    "今の現場では、2メートル近いケーブルのジョイントの組立図を、わら半紙で壁に貼っています。佐藤くんは壁を見に行って、戻ったら寸法を忘れてしまう。ARグラスなら、実物大の完成形がケーブルの横に出ます。過去にミスがあった所では、赤を切るか青を切るかのように、声とレーザーで止めてくれます。火花や異物から目も守ります。ただし、寸法の最終確認はスケールで。測る道具ではなく、間違いに気づく道具です。"],
  6: ["第4話・40秒", "会う → 試す → 決める → 広がる",
    "最初のお客様は、関東の中堅の電気工事会社です。中でも、電力会社や官公庁の工事が多く、写真台帳と報告書が多い会社を狙います。業界の勉強会で、2024年からの残業の上限と書類の削減について話します。展示会では、デモを自分の現場に当てはめて試してもらいます。次に、3か月の有償の試験導入で、報告書の時間が本当に減るかを数字で確かめます。本契約になったら、効果が出た班から台数を増やし、協力会社にも同じ形で使ってもらいます。"],
  7: ["お金の話1・40秒", "会社の基盤 月5万円＋メガネ1台 月2万円。10台で年300万円",
    `では、誰が何にいくら払うのか。払うのは電気工事会社です。1拠点、つまり1つの支店や営業所で、会社のデータ置き場や管理画面の基盤に月5万円、メガネ1台ごとに月2万円。標準の10台なら月25万円、年300万円です。報告書の形に整えるところまで、この中に入っています。メガネ本体は、お客様の名前でリースし、手続きは私たちが代わりにします。10台で年約${man(V.leaseSite())}万円です。初年度だけ、私たちが2か月現場に入る導入支援に200万円いただきます。スマホの契約と同じで、初期設定は最初だけ、月額は毎月です。`],
  8: ["お金の話2・40秒", `1台1日35分減ると年${man(CV.docs)}万円。${be}分減れば元が取れる`,
    `なぜ工事部長は払うのか。メガネを掛けた班では、職長の日報と撮影が1日15分、施工管理者の写真整理と写真台帳が20分減ると考えています。10台を年240日使うと、浮く時間の価値は年約${man(CV.docs)}万円。1時間の人件費は、国の統計の年収に会社が払う社会保険料を足し、残業を含む年${man(V.WAGE.annualHours)}時間で割った約${man(Math.round(HAVG * 1000) * 10)}円です。お客様が払うのは利用料とリースで年約${man(CV.cost)}万円なので、約${CV.ratio.toFixed(1)}倍。逆に言えば、1台1日${be}分減れば元が取れます。これを実証の合格ラインにします。`],
  9: ["お金の話2つづき・40秒", "計画に入れるのは確かな分だけ。手戻りは測れたら上乗せ",
    "工事部長にとっての効果は、書類の時間だけではありません。やり直しが減れば、ARが効く工種だけでも年約360万円。ただ、これはAIの出来と私たちの監修次第なので、実証で測れるまで計画には入れません。新人の指導は、他の仕事と混ざっていて測れないので、金額にせず、独り立ちまでの月数で測ります。確かな分だけで約" + CV.ratio.toFixed(1) + "倍、手戻りも減れば約" + CV.ratioUp.toFixed(1) + "倍です。"],
  10: ["お金の話3・40秒", "利用料300万円のうち216万円が残る。だから利用料を積み上げる",
    "次に、当社の財布に何が残るか。300万円の利用料のうち、AIの文字起こしや画像の解析、動画の保存の費用が、メガネ1台あたり月約6,000円、会社ごとのデータ置き場が月1万円で、10台なら年84万円。216万円が残ります。導入支援の200万円は、担当者が42人日現場に入り、交通費もかかるので、残るのは約59万円。だから当社は、人が動く仕事ではなく、利用料を積み上げる会社にします。"],
  11: ["お金の話4・40秒", "社員1人で約4拠点を見れば黒字。10台のままでは5年目でも届かない",
    `なぜ赤字が続くのか。社員1人にかかる費用は、給料と会社負担、採用費、家賃などを合わせて年約${man(PER)}万円。1拠点から残るのは${man(U.gpYear)}万円。割ると約${(PER / U.gpYear).toFixed(1)}。新しいお客様は最初の年は半年分しか払わないので、年度末に数えると1人約4拠点が目安です。社員は、営業だけでなく技術や導入コンサルタントも含めた全員です。人を先に雇うので、10台のままでは5年目でも${Y5.sitesPerStaff.toFixed(2)}で届きません。だから、効果が出た班からメガネを増やしてもらいます。`],
  12: ["お金の話5・40秒", "去年のお客様が今年も払う。売上は雪だるま式",
    `利用料は毎年いただくので、去年までのお客様の分が今年の売上に乗ります。解約は毎年1割と控えめに見ています。5年目の売上約${man(Y5.sales / 100)}百万円のうち、${Math.round(Y5.existing / Y5.sales * 100)}%は前の年までのお客様からの利用料です。新しいお客様が増えるたびに、雪だるまのように大きくなります。10台のままでは5年目もまだ赤字で、1拠点15台に増やせれば${F15}年目に黒字です。10年で見ると、10台のままでも${R10.firstBlack}年目に黒字になり、${R10.cumBack}年目に5年間の赤字を取り戻します。`],
  13: ["お金の話6・30秒", `${man(U.cac)}万円かけて連れてきたお客様が、${man(U.ltv(7))}万円残してくれる`,
    `お客様1拠点を連れてくる費用は約${man(U.cac)}万円。営業3人の人件費と広告費を、新しいお客様の数で割った数字です。一方、1拠点が7年間で残してくれる粗利は約${man(U.ltv(7))}万円。約${Math.round(U.ltv(7) / U.cac)}倍です。${man(U.cac)}万円は約${Math.round(U.paybackMonths)}か月で回収できます。解約が年1割なら平均10年続くので、7年は控えめに見た数字です。`],
  14: ["お金の話7・40秒", "黒字化の条件は、1拠点の台数。止まる線も先に決める",
    `うまくいかなかったらどうするか。計算すると、1拠点10台のままだと、計画どおりにお客様が増えても5年以内に黒字になりません。でも、効果が出た班から台数を増やして15台にできれば、新しいお客様が計画の6割でも5年目に黒字です。だから大事なのは、拠点の数より、1拠点で使ってもらう台数です。そして止まる線を先に決めます。実証で1台1日${be}分減れば進む、15分未満なら撤退。1年目の終わりに、試してくれた3社のうち本契約が0社なら撤退か方向転換。2年目の終わりに平均12台に届かなければ、採用を止めます。`],
  15: ["準備期・30秒", "準備期の6か月で、最初の1拠点を動かす",
    `最初の6か月は準備期です。費用は約500万円で、自己資金でまかないます。私の給料はゼロです。3か月で試作品、6か月で1拠点の実証。1台1日${be}分減ったか、メガネで1日記録できるかを確かめます。合格したら、自己資金の2億円を入れて、仲間を採用します。`],
  16: ["全体図・30秒", "上の矢印は記録とサービス、下の矢印はお金",
    "1枚にまとめるとこうなります。上の流れが記録とサービス。お客様の現場から記録が来て、照合・報告書・教材にして返す。下の流れがお金。お客様から10台で年300万円が入り、84万円がAIと保存の費用、216万円が残って社員の費用をまかなう。社員1人で約4拠点を超えたら黒字。そして記録が貯まるほど、他社がすぐには真似できなくなります。"],
  17: ["ツッコミ対策・50秒", "4つのツッコミには、この一言で返す",
    "よく聞かれる4つの質問への答えです。本当に300万円も払うのか。1台1日" + be + "分減れば元が取れます。想定は35分で、実証で測ります。なぜ赤字が続くのか。人を先に雇うからで、社員1人約4拠点を超えたら黒字になります。だから台数を増やしてもらいます。真似されないか。現場の記録は使うほど貯まり、後から来た会社には移せません。日立と同じではないか。日立は自社の保守契約の中で、安全や点検にAIを使っています。当社は中堅の電気工事会社の施工の書類と若手の育成に絞り、月額で提供します。"],
  18: ["まとめ・30秒", "300万・216万・4。この3つで話せる",
    `まとめです。お客様が10台で年300万円払う。216万円が残る。社員1人約4拠点で黒字。この3つの数字で、すべて説明できます。10年で見ると、10台のままでも${R10.firstBlack}年目に黒字になります。詳しい計算の過程は、別冊の計算詳細版にまとめています。以上で終わります。ご清聴ありがとうございました。`],
};
const NOTES = Object.fromEntries(Object.entries(NOTE_PARTS).map(([n, [t, msg, body]]) => [n, `【${t}】\n要約：${msg}\n\n${body}`]));

async function main() {
  const pres = new pptxgen();
  const add0 = await L.setup(pres, "マンガでわかる カンツギのもうけの仕組み", NOTES);
  // 表題欄の左に章を書く
  const CHAPTER = [[2, "登場人物"], [3, "現場の物語"], [7, "お金の話"], [15, "準備期"], [16, "まとめ"]];
  const add = (n) => {
    const s = add0(n);
    const c = n > 1 && CHAPTER.filter(([p]) => p <= n).pop();
    if (c) txt(s, c[1], { x: 6.62, y: 5.205, w: 1.28, h: 0.29, size: 7.5, color: L.NAVY, bold: true, align: "center", valign: "middle" });
    return s;
  };

  const C = {};
  for (const r of ["veteran", "junior", "manager", "founder", "client"])
    for (const f of ["normal", "happy", "worried", "shock", "think"]) C[r + "-" + f] = await png(charSVG(r, f), 300);
  const I = {};
  for (const k of ["office", "coin", "server", "film", "report", "glasses", "clock", "book", "bait", "fish"]) I[k] = await png(iconSVG(k), 300);
  const ch = (s, key, x, y, h) => s.addImage({ data: C[key], x, y, w: h * 200 / 240, h });
  const ic = (s, key, x, y, w) => s.addImage({ data: I[key], x, y, w, h: w });

  // ---------- 1 表紙 ----------
  {
    const s = add(1);
    s.addImage({ data: await png("kantsugi-logotype-color"), x: X0, y: 0.3, w: 1.7, h: 1.7 * 120 / 560 });
    txt(s, "事業計画書　やさしい版", { x: X0 + 1.85, y: 0.3, w: 3.5, h: 0.36, size: 11, color: K.gray700, valign: "middle" });
    txt(s, [run("マンガでわかる", { breakLine: true }), run("カンツギの"), run("もうけの仕組み", { color: ACC })], { x: X0, y: 0.68, w: 6.0, h: 1.1, size: 28, bold: true, lsm: 1.1 });
    box(s, { x: 6.75, y: 0.45, w: 2.85, h: 1.25, line: ACC, lw: 2, name: "pass-line" });
    txt(s, "元が取れる条件", { x: 6.95, y: 0.52, w: 2.5, h: 0.28, size: 11, bold: true, color: K.gray700 });
    txt(s, `1台1日${Math.round(CV.breakEvenMin)}分`, { x: 6.95, y: 0.8, w: 2.6, h: 0.8, size: 24, bold: true, color: ACC, valign: "middle" });
    const nums = [["お客様が払う", "300万円", "1拠点・10台\n1年あたり"], ["当社に残る", `${man(U.gpYear)}万円`, "AI・保存の\n費用を引いた粗利"], ["黒字のライン", "約4拠点", "社員1人あたり"]];
    nums.forEach(([a, b, c], i) => {
      const x = X0 + i * 3.1, y = 2.0;
      box(s, { x, y, w: 2.95, h: 1.0, fill: i === 1 ? K.ink : K.fill, name: "num-" + i });
      const col = i === 1 ? K.paper : K.ink;
      txt(s, [run(String(i + 1) + "  ", { color: ACC, bold: true }), run(a)], { x: x + 0.15, y: y + 0.08, w: 2.7, h: 0.28, size: 11, bold: true, color: col });
      txt(s, b, { x: x + 0.15, y: y + 0.36, w: 1.55, h: 0.55, size: 22, bold: true, color: i === 1 ? L.YELLOW : ACC, valign: "middle" });
      txt(s, c, { x: x + 1.65, y: y + 0.4, w: 1.25, h: 0.5, size: 8, color: col, valign: "middle" });
    });
    const people = [["veteran-normal", "熟練者"], ["junior-normal", "若手"], ["manager-worried", "工事部長"], ["founder-happy", "社長（わたし）"]];
    people.forEach(([k, n], i) => {
      const x = X0 + 0.1 + i * 1.25;
      ch(s, k, x, 3.25, 1.5);
      txt(s, n, { x: x - 0.1, y: 4.78, w: 1.45, h: 0.24, size: 9, align: "center", color: K.gray700 });
    });
    bubble(s, "数字は3つだけ。\nこれで全部、説明できます！", { x: 5.6, y: 3.3, w: 3.4, h: 0.95, tail: "bl", size: 13, bold: true, name: "cover-bubble" });
    box(s, { x: 6.6, y: 5.2, w: 3.28, h: 0.305, line: L.NAVY });
    txt(s, [run("件名 ", { color: K.gray700 }), run("カンツギ 事業計画書 やさしい版", { bold: true, color: L.NAVY })], { x: 6.68, y: 5.2, w: 2.6, h: 0.305, size: 8, valign: "middle" });
    txt(s, "1／18", { x: 9.3, y: 5.2, w: 0.58, h: 0.305, size: 9, bold: true, color: L.NAVY, align: "center", valign: "middle" });
    s.addShape("line", { x: 9.3, y: 5.2, w: 0, h: 0.305, line: { color: L.NAVY, width: 0.75 } });
  }

  // ---------- 2 登場人物 ----------
  {
    const s = add(2);
    heading(s, "登場人物 ─ お金を払うのは「会社」。窓口は工事部長", "この4人で、困りごと → 解決 → お金の流れを追いかける");
    const cast = [
      ["veteran-normal", "田中さん（62歳）", "熟練の電工。来年、定年", "勘で分かる。\nでも説明はできん"],
      ["junior-worried", "佐藤くん（24歳）", "入社2年目の施工管理", "田中さんが辞めたら\nどうしよう…"],
      ["manager-worried", "山本工事部長（50歳）", "管理者。お金を払う人", "報告書と育成で\n手一杯だ"],
      ["founder-happy", "社長（わたし）", "カンツギ。元・施工管理", "現場の勘を、\n次の担い手へ"],
    ];
    const cw = 2.2, gap = 0.133, top = 1.15;
    cast.forEach(([k, n, r, say], i) => {
      const x = X0 + i * (cw + gap);
      const payer = i === 2;
      box(s, { x, y: top, w: cw, h: 3.45, line: payer ? ACC : K.ink, lw: payer ? 2.5 : 1.25, name: "cast-" + i });
      if (payer) { box(s, { x: x + cw - 1.05, y: top, w: 1.05, h: 0.26, fill: ACC }); txt(s, "お金を払う人", { x: x + cw - 1.05, y: top, w: 1.05, h: 0.26, size: 8.5, bold: true, color: K.paper, align: "center", valign: "middle" }); }
      ch(s, k, x + (cw - 1.25) / 2, top + 0.2, 1.5);
      txt(s, n, { x: x + 0.1, y: top + 1.75, w: cw - 0.2, h: 0.3, size: 12, bold: true, align: "center" });
      txt(s, r, { x: x + 0.1, y: top + 2.05, w: cw - 0.2, h: 0.26, size: 9, color: K.gray700, align: "center" });
      bubble(s, say, { x: x + 0.15, y: top + 2.5, w: cw - 0.3, h: 0.78, tail: "tl", size: 10 });
    });
    box(s, { x: X0, y: 4.72, w: W, h: 0.38, fill: K.fill });
    txt(s, [run("ポイント：", { bold: true, color: ACC }), run("使うのは現場、払うのは会社。だから「会社にとって効果があるか」で値段を決める")], { x: X0 + 0.15, y: 4.72, w: W - 0.3, h: 0.38, size: 10.5, valign: "middle" });
  }

  // ---------- 3 第1話 困りごと ----------
  {
    const s = add(3);
    heading(s, "第1話　このままだと、現場が止まる", "熟練者の勘は手順書に書かれていない。辞めたら消える");
    const pw = 2.2, gap = 0.133, top = 1.12, ph = 3.5;
    const P = (i) => X0 + i * (pw + gap);
    panel(s, { x: P(0), y: top, w: pw, h: ph, cap: "起　ある日の工事現場" });
    ch(s, "veteran-think", P(0) + 0.45, top + 1.6, 1.6);
    bubble(s, "端末の寸法は\nテープの巻き始め\nから測るんじゃ", { x: P(0) + 0.12, y: top + 0.45, w: pw - 0.24, h: 0.85, tail: "bl", size: 10.5 });

    panel(s, { x: P(1), y: top, w: pw, h: ph, cap: "承　若手は分からない" });
    ch(s, "junior-shock", P(1) + 0.1, top + 1.85, 1.35);
    ch(s, "veteran-normal", P(1) + 1.15, top + 2.05, 1.15);
    bubble(s, "えっ、なんで\n分かるんですか！？", { x: P(1) + 0.1, y: top + 0.45, w: pw - 0.2, h: 0.8, tail: "bl", size: 10.5 });
    bubble(s, "勘じゃ", { x: P(1) + 1.2, y: top + 1.42, w: 0.85, h: 0.45, tail: "br", size: 10 });

    panel(s, { x: P(2), y: top, w: pw, h: ph, cap: "転　工事部長、青ざめる" });
    ch(s, "manager-shock", P(2) + 0.45, top + 1.75, 1.55);
    bubble(s, "田中さんは来年定年…\n手順書に書いてない！", { x: P(2) + 0.1, y: top + 0.45, w: pw - 0.2, h: 0.85, tail: "bl", size: 10 });
    sfx(s, "ガーン", { x: P(2) + 1.15, y: top + 1.35, size: 20 });

    panel(s, { x: P(3), y: top, w: pw, h: ph, cap: "結　しかも時間がない", fill: K.fill });
    ch(s, "manager-worried", P(3) + 0.15, top + 1.85, 1.45);
    ic(s, "report", P(3) + 1.35, top + 2.1, 0.6);
    ic(s, "clock", P(3) + 1.4, top + 2.75, 0.5);
    bubble(s, "工事写真と報告書で\n毎晩残業。教える\n時間もない…", { x: P(3) + 0.1, y: top + 0.42, w: pw - 0.2, h: 1.05, tail: "bl", size: 10 });
    box(s, { x: X0, y: 4.72, w: W, h: 0.38, fill: K.ink });
    txt(s, [run("現実も同じ：", { bold: true, color: L.YELLOW }), run("建設業で働く人の36.7%が55歳以上、29歳以下は11.7%（国交省、2024年）。2024年4月から残業にも上限")], { x: X0 + 0.15, y: 4.72, w: W - 0.3, h: 0.38, size: 9, color: K.paper, valign: "middle" });
  }

  // ---------- 4 第2話 解決 ----------
  {
    const s = add(4);
    heading(s, "第2話　メガネを掛けて、いつも通り働くだけ", "動画と会話をAIが整理して、3つの形で渡す");
    const pw = 2.2, gap = 0.133, top = 1.12, ph = 3.5;
    const P = (i) => X0 + i * (pw + gap);
    panel(s, { x: P(0), y: top, w: pw, h: ph, cap: "① 掛けるだけ" });
    ic(s, "glasses", P(0) + 0.5, top + 1.3, 1.2);
    ch(s, "veteran-happy", P(0) + 0.55, top + 2.2, 1.15);
    bubble(s, "いつも通り\nやるだけでええんか？", { x: P(0) + 0.1, y: top + 0.42, w: pw - 0.2, h: 0.78, tail: "bl", size: 10 });

    panel(s, { x: P(1), y: top, w: pw, h: ph, cap: "② AIが整理" });
    ic(s, "film", P(1) + 0.2, top + 0.6, 0.8);
    arrow(s, P(1) + 1.05, top + 1.0, 0.35, 0);
    ic(s, "server", P(1) + 1.35, top + 0.6, 0.75);
    txt(s, "動画と会話", { x: P(1) + 0.05, y: top + 1.42, w: 1.1, h: 0.25, size: 9, align: "center" });
    txt(s, "AIが整理", { x: P(1) + 1.1, y: top + 1.42, w: 1.1, h: 0.25, size: 9, align: "center" });
    sfx(s, "ピコン！", { x: P(1) + 0.25, y: top + 1.75, size: 20 });
    txt(s, [run("「どの場面で」", { breakLine: true }), run("「何を見て」", { breakLine: true }), run("「どう判断したか」", { breakLine: true }), run("を自動で切り出す")], { x: P(1) + 0.15, y: top + 2.35, w: pw - 0.3, h: 1.05, size: 10, align: "center" });

    panel(s, { x: P(2), y: top, w: pw, h: ph, cap: "③ 若手へ：照合" });
    ch(s, "junior-happy", P(2) + 0.45, top + 1.85, 1.5);
    bubble(s, "図面と8mmズレてる\nって、その場で\n教えてくれた！", { x: P(2) + 0.1, y: top + 0.42, w: pw - 0.2, h: 1.05, tail: "bl", size: 10 });

    panel(s, { x: P(3), y: top, w: pw, h: ph, cap: "④ 管理者へ：報告書", fill: K.fill });
    ch(s, "manager-happy", P(3) + 0.1, top + 1.85, 1.45);
    ic(s, "report", P(3) + 1.38, top + 1.95, 0.6);
    ic(s, "book", P(3) + 1.35, top + 2.65, 0.65);
    bubble(s, "報告書が勝手に\nできてる！教材にも\nなるのか", { x: P(3) + 0.1, y: top + 0.42, w: pw - 0.2, h: 1.05, tail: "bl", size: 10 });
    box(s, { x: X0, y: 4.72, w: W, h: 0.38, fill: K.fill });
    txt(s, [run("渡す先は3つ：", { bold: true, color: ACC }), run("若手へ（図面と照合）　管理者へ（報告書を自動作成）　あとから（教材・研修資料）")], { x: X0 + 0.15, y: 4.72, w: W - 0.3, h: 0.38, size: 10.5, valign: "middle" });
  }

  // ---------- 5 第3話 図面が手元に ----------
  {
    const s = add(5);
    heading(s, "第3話　壁の図面が、手元に出る", "約2mのケーブルジョイント。わら半紙の組立図を見に行く往復をなくす");
    const pw = 2.2, gap = 0.133, top = 1.12, ph = 3.5;
    const P = (i) => X0 + i * (pw + gap);
    panel(s, { x: P(0), y: top, w: pw, h: ph, cap: "① いつもの現場" });
    box(s, { x: P(0) + 1.25, y: top + 1.55, w: 0.8, h: 1.0, fill: L.PALEY, line: K.gray500 });
    txt(s, "組立図\n（壁）", { x: P(0) + 1.25, y: top + 1.55, w: 0.8, h: 1.0, size: 8.5, align: "center", valign: "middle", color: K.gray700 });
    ch(s, "junior-worried", P(0) + 0.1, top + 2.0, 1.4);
    bubble(s, "壁の図面を見に行って\n戻ったら寸法を\n忘れた…", { x: P(0) + 0.1, y: top + 0.42, w: pw - 0.2, h: 1.0, tail: "bl", size: 9.5 });
    panel(s, { x: P(1), y: top, w: pw, h: ph, cap: "② ARで手元に" });
    [[0, 0.3, K.gray700], [0.3, 0.7, K.rule], [0.7, 1, K.gray700]].forEach(([a, b, c]) => box(s, { x: P(1) + 0.15 + a * 1.9, y: top + 2.95, w: (b - a) * 1.9, h: 0.2, fill: c }));
    s.addShape("rect", { x: P(1) + 0.15 + 0.3 * 1.9, y: top + 2.8, w: 0.4 * 1.9, h: 0.5, fill: { color: K.paper, transparency: 100 }, line: { color: ACC, width: 1.5, dashType: "dash" } });
    ch(s, "junior-happy", P(1) + 0.6, top + 1.55, 1.15);
    bubble(s, "実物大の完成形が\nケーブルの横に\n出た！", { x: P(1) + 0.1, y: top + 0.42, w: pw - 0.2, h: 1.0, tail: "bl", size: 9.5 });
    panel(s, { x: P(2), y: top, w: pw, h: ph, cap: "③ 赤を切るか、青か" });
    s.addShape("line", { x: P(2) + 0.3, y: top + 2.3, w: 1.6, h: 0, line: { color: L.RED, width: 5 } });
    s.addShape("line", { x: P(2) + 0.3, y: top + 2.65, w: 1.6, h: 0, line: { color: "2F6DB5", width: 5 } });
    txt(s, "✂", { x: P(2) + 0.85, y: top + 2.85, w: 0.5, h: 0.45, size: 22, align: "center" });
    bubble(s, "ここは過去にミスあり。\n赤を切れ", { x: P(2) + 0.1, y: top + 0.42, w: pw - 0.2, h: 0.9, tail: "br", size: 10, bold: true, shout: true });
    sfx(s, "ピピッ", { x: P(2) + 0.25, y: top + 1.45, size: 18 });
    panel(s, { x: P(3), y: top, w: pw, h: ph, cap: "④ 目も守る", fill: K.fill });
    ch(s, "veteran-happy", P(3) + 0.45, top + 1.85, 1.5);
    sfx(s, "バチッ", { x: P(3) + 1.1, y: top + 1.4, size: 16 });
    bubble(s, "火花が飛んでも\n目は守られとる", { x: P(3) + 0.1, y: top + 0.42, w: pw - 0.2, h: 0.85, tail: "bl", size: 10 });
    box(s, { x: X0, y: 4.72, w: W, h: 0.38, fill: K.ink });
    txt(s, [run("約束：", { bold: true, color: L.YELLOW }), run("寸法の最終確認はスケールで。ARは「測る道具」ではなく「理解して、間違いに気づく道具」")], { x: X0 + 0.15, y: 4.72, w: W - 0.3, h: 0.38, size: 10, color: K.paper, valign: "middle" });
  }

  // ---------- 6 第4話 どう会う ----------
  {
    const s = add(6);
    heading(s, "第4話　最初のお客様に、どう会う？", "関東の中堅電気工事会社（施工管理者30人以上）から。会う → 試す → 決める → 広がる");
    const pw = 2.2, gap = 0.133, top = 1.12, ph = 3.5;
    const P = (i) => X0 + i * (pw + gap);
    panel(s, { x: P(0), y: top, w: pw, h: ph, cap: "① 勉強会で話す" });
    ch(s, "founder-happy", P(0) + 0.45, top + 1.75, 1.55);
    bubble(s, "2024年からの残業上限、\n書類の時間を\n減らしませんか？", { x: P(0) + 0.1, y: top + 0.42, w: pw - 0.2, h: 1.0, tail: "bl", size: 9.5 });
    panel(s, { x: P(1), y: top, w: pw, h: ph, cap: "② 展示会で試す" });
    ic(s, "glasses", P(1) + 0.55, top + 1.45, 1.1);
    ch(s, "junior-happy", P(1) + 0.55, top + 2.2, 1.15);
    bubble(s, "自分の現場で\n試してみたい！", { x: P(1) + 0.1, y: top + 0.42, w: pw - 0.2, h: 0.8, tail: "bl", size: 10 });
    panel(s, { x: P(2), y: top, w: pw, h: ph, cap: "③ 3か月の試験導入" });
    ch(s, "manager-think", P(2) + 0.45, top + 1.8, 1.5);
    bubble(s, "報告書の時間が\n本当に減るか、\n数字で見たい", { x: P(2) + 0.1, y: top + 0.42, w: pw - 0.2, h: 1.0, tail: "bl", size: 10 });
    sfx(s, "実測！", { x: P(2) + 1.2, y: top + 1.45, size: 18 });
    panel(s, { x: P(3), y: top, w: pw, h: ph, cap: "④ 本契約と紹介", fill: K.fill });
    ch(s, "manager-happy", P(3) + 0.1, top + 1.95, 1.3);
    ch(s, "client-happy", P(3) + 1.15, top + 2.1, 1.15);
    bubble(s, "協力会社にも\n同じ形で使って\nもらおう", { x: P(3) + 0.1, y: top + 0.42, w: pw - 0.2, h: 1.0, tail: "bl", size: 10 });
    box(s, { x: X0, y: 4.72, w: W, h: 0.38, fill: K.ink });
    txt(s, [run("5年の見込み：", { bold: true, color: L.YELLOW }), run(`条件に合う約180社を中心に5年で約85拠点（解約を引いて${Y5.sites}）。効果が出た班から台数を増やす`)], { x: X0 + 0.15, y: 4.72, w: W - 0.3, h: 0.38, size: 9.5, color: K.paper, valign: "middle" });
  }

  // ---------- 5 お金の話1 誰が何にいくら ----------
  {
    const s = add(7);
    heading(s, "お金の話1　誰が、何に、いくら払う？", "払うのは電気工事会社。会社の基盤＋メガネの台数。10台で年300万円、初年度だけ導入支援200万円");
    const top = 1.2;
    ic(s, "office", X0 + 0.15, top + 0.3, 1.1);
    ch(s, "manager-normal", X0 + 0.25, top + 1.6, 1.15);
    txt(s, "電気工事会社（お客様）", { x: X0 - 0.25, y: top + 2.8, w: 1.95, h: 0.26, size: 10, bold: true, align: "center" });
    ch(s, "founder-happy", X0 + 3.6, top + 1.0, 1.55);
    txt(s, "カンツギ", { x: X0 + 3.35, y: top + 2.6, w: 1.8, h: 0.26, size: 10, bold: true, align: "center" });
    arrow(s, X0 + 1.55, top + 1.0, 1.9, 0, { lw: 4, color: ACC });
    ic(s, "coin", X0 + 2.15, top + 0.25, 0.6);
    txt(s, [run("年300万円", { bold: true, fontSize: 16, color: ACC, breakLine: true }), run("（10台・月25万円）", { fontSize: 10 })], { x: X0 + 1.5, y: top + 1.12, w: 2.0, h: 0.6, align: "center" });
    arrow(s, X0 + 1.55, top + 2.1, 1.9, 0, { lw: 2, color: K.gray500, dash: "dash" });
    txt(s, "初年度だけ ＋200万円\n（導入支援）", { x: X0 + 1.5, y: top + 2.18, w: 2.0, h: 0.5, size: 9.5, align: "center", color: K.gray700 });
    const rx = 5.55, rw = 4.05;
    box(s, { x: rx, y: top - 0.05, w: rw, h: 3.0, line: K.ink, lw: 1.25, name: "receipt" });
    txt(s, "御請求書（1拠点・1年分）", { x: rx + 0.2, y: top + 0.05, w: rw - 0.4, h: 0.32, size: 12, bold: true, align: "center" });
    hline(s, rx + 0.2, top + 0.42, rw - 0.4, K.ink, 1.5);
    const lines = [["会社の基盤", "データ置き場・管理画面 月5万円×12", "60万円"], ["メガネの利用料", "1台 月2万円×10台×12", "240万円"]];
    lines.forEach(([a, b, c], i) => {
      const y = top + 0.5 + i * 0.55;
      txt(s, [run(a, { bold: true, breakLine: true }), run(b, { fontSize: 8.5, color: K.gray700 })], { x: rx + 0.2, y, w: 2.6, h: 0.5, size: 10.5 });
      txt(s, c, { x: rx + 2.8, y, w: 1.05, h: 0.5, size: 11, align: "right", valign: "middle" });
    });
    hline(s, rx + 0.2, top + 1.65, rw - 0.4, K.ink, 1.5);
    txt(s, "合計（毎年）", { x: rx + 0.2, y: top + 1.72, w: 2, h: 0.4, size: 12, bold: true, valign: "middle" });
    txt(s, "300万円", { x: rx + 2.2, y: top + 1.72, w: 1.65, h: 0.4, size: 18, bold: true, color: ACC, align: "right", valign: "middle" });
    s.addShape("line", { x: rx + 0.2, y: top + 2.2, w: rw - 0.4, h: 0, line: { color: K.gray500, width: 1, dashType: "dash" } });
    txt(s, [run("初年度だけ　導入支援 200万円", { bold: true, breakLine: true }), run(`メガネ本体はお客様の名前でリース、手続きは当社（10台で年約${man(V.leaseSite())}万円）`, { fontSize: 8.5, color: K.gray700 })], { x: rx + 0.2, y: top + 2.27, w: rw - 0.4, h: 0.6, size: 10 });
    box(s, { x: X0, y: 4.45, w: W, h: 0.62, fill: K.fill });
    txt(s, [run("たとえるなら、スマホの契約と同じ。", { bold: true, color: ACC, breakLine: true }), run("初期設定（導入支援）は最初だけ。月額料金（利用料）は毎月入ってくる。台数が増えれば、そのぶん月額も増える")], { x: X0 + 0.15, y: 4.45, w: W - 0.3, h: 0.62, size: 10.5, valign: "middle" });
  }

  // ---------- 6 お金の話2 なぜ払う ----------
  {
    const s = add(8);
    heading(s, "お金の話2　工事部長は、なぜ払うの？", `メガネを掛けた班の書類の時間が減る。1台1日${Math.round(CV.breakEvenMin)}分減れば元が取れる`);
    const top = 1.2;
    const eq = [["10台", "メガネの数"], ["×", ""], ["240日", "1年に使う日"], ["×", ""], ["35分", "1台1日減る時間"], ["×", ""], [`約${man(Math.round(HAVG * 1000) * 10)}円`, "1時間の人件費"], ["＝", ""], [`年${man(CV.docs)}万円`, "浮く時間の価値"]];
    let x = X0;
    eq.forEach(([a, b], i) => {
      const op = b === "", last = i === eq.length - 1;
      const w = op ? 0.3 : last ? 1.75 : 1.2;
      if (!op) box(s, { x, y: top, w, h: 0.9, fill: last ? K.ink : K.fill });
      txt(s, a, { x, y: top + (op ? 0.15 : 0.08), w, h: op ? 0.6 : 0.5, size: op ? 20 : 15, bold: true, align: "center", valign: "middle", color: last ? L.YELLOW : K.ink });
      if (!op) txt(s, b, { x, y: top + 0.56, w, h: 0.28, size: 8, align: "center", color: last ? K.paper : K.gray700 });
      x += w + 0.04;
    });
    const by = top + 1.15, bx = X0 + 1.3, scale = 3.6 / 480;
    txt(s, "1年あたり（2年目以降）", { x: X0, y: by, w: 3, h: 0.26, size: 10, bold: true });
    [["払うお金", CV.cost, K.gray500, `年${man(CV.cost)}万円`], ["浮くお金", CV.docs, ACC, `年${man(CV.docs)}万円`]].forEach(([n, v, c, lab], i) => {
      const y = by + 0.33 + i * 0.5;
      txt(s, n, { x: X0, y, w: 1.3, h: 0.4, size: 11, bold: true, valign: "middle" });
      box(s, { x: bx, y, w: v * scale, h: 0.4, fill: c });
      txt(s, lab, { x: bx + v * scale + 0.08, y, w: 3.0, h: 0.4, size: 10, valign: "middle" });
    });
    ch(s, "manager-happy", 8.3, top + 1.15, 1.35);
    bubble(s, `1日${Math.round(CV.breakEvenMin)}分減れば\n元が取れるのか！`, { x: 6.55, y: top + 1.0, w: 1.75, h: 0.7, tail: "br", size: 10.5, bold: true });
    box(s, { x: X0, y: 3.9, w: W, h: 1.18, fill: K.fill });
    txt(s, [
      run("元が取れる条件：", { bold: true, color: ACC }), run(`払うお金${man(CV.cost)}万円 ÷（10台×240日）÷ 1時間の人件費 ＝ 1台1日約${Math.round(CV.breakEvenMin)}分`, { bold: true, breakLine: true }),
      run("35分って？", { bold: true, color: ACC }), run("　職長の日報・撮影15分＋施工管理者の写真整理・写真台帳20分（実証で測る）", { breakLine: true }),
      run("1時間の人件費って？", { bold: true, color: ACC }), run(`　国の統計の年収×会社負担${V.WAGE.employerLoad}÷残業を含む年${man(V.WAGE.annualHours)}時間（電工約${man(Math.round(V.H.electrician * 100) * 100)}円・施工管理約${man(Math.round(V.H.manager * 100) * 100)}円）`, { breakLine: true }),
      run("初年度は？", { bold: true, color: ACC }), run(`　導入支援200万円も払うので、約${Math.round((CV.cost + 200) / (CV.docs / 12))}か月で元が取れる`),
    ], { x: X0 + 0.15, y: 3.9, w: W - 0.3, h: 1.18, size: 9.5, valign: "middle", psa: 1 });
  }

  // ---------- 9 お金の話2つづき 3つの効果 ----------
  {
    const s = add(9);
    heading(s, "お金の話2　つづき ─ 計画に入れるのは、確かな分だけ", `確かな分（書類の時間）で約${CV.ratio.toFixed(1)}倍。手戻りも減れば約${CV.ratioUp.toFixed(1)}倍`);
    const top = 1.15;
    const items = [["clock", CV.levers[0], "計画に入れる"], ["report", CV.levers[1], "測れたら上乗せ"], ["book", CV.levers[2], "金額にしない"]];
    items.forEach(([k, l, tag], i) => {
      const x = X0 + i * 2.05;
      box(s, { x, y: top, w: 1.9, h: 2.1, fill: i === 0 ? K.fill : K.paper, line: i === 0 ? undefined : K.rule });
      ic(s, k, x + 0.6, top + 0.12, 0.7);
      txt(s, l.name, { x: x + 0.08, y: top + 0.86, w: 1.74, h: 0.3, size: 10, bold: true, align: "center" });
      txt(s, l.value ? `年${man(l.value)}万円` : "月数で測る", { x: x + 0.08, y: top + 1.16, w: 1.74, h: 0.36, size: 14, bold: true, color: i === 0 ? ACC : K.gray700, align: "center", valign: "middle" });
      txt(s, tag, { x: x + 0.08, y: top + 1.52, w: 1.74, h: 0.2, size: 8.5, bold: true, align: "center", color: i === 0 ? ACC : K.gray700 });
      txt(s, l.formula, { x: x + 0.08, y: top + 1.72, w: 1.74, h: 0.36, size: 7, color: K.gray700, align: "center" });
    });
    box(s, { x: X0, y: top + 2.25, w: 6.05, h: 0.75, fill: K.ink });
    txt(s, [run("計画に入れる効果 ", { color: K.paper }), run(`年${man(CV.docs)}万円`, { color: L.YELLOW, fontSize: 20 }), run(`　払うのは${man(CV.cost)}万円 → 約${CV.ratio.toFixed(1)}倍`, { color: K.paper, fontSize: 11 })], { x: X0 + 0.15, y: top + 2.25, w: 5.8, h: 0.75, size: 12, bold: true, valign: "middle" });
    ch(s, "manager-happy", 7.9, 1.6, 1.55);
    bubble(s, "確かな分だけで\n元が取れるなら\n安心だ", { x: 6.7, y: 1.12, w: 1.6, h: 0.95, tail: "br", size: 9.5 });
    box(s, { x: 6.7, y: 3.25, w: 2.9, h: 0.9, line: ACC, lw: 1.5 });
    txt(s, [run(`浮く時間 年${man(CV.freedHours)}時間`, { bold: true, breakLine: true }), run("10台分。掛けた班の時間だけを数える", { fontSize: 9 })], { x: 6.8, y: 3.25, w: 2.7, h: 0.9, size: 10.5, valign: "middle" });
    box(s, { x: X0, y: 4.45, w: W, h: 0.5, fill: K.fill });
    txt(s, [run("大きく言わない：", { bold: true, color: ACC }), run("効果を数えるのは全員ではなく、メガネを掛けた班だけ。手戻りはAIの出来と当社の監修しだいなので、測れてから足す")], { x: X0 + 0.15, y: 4.45, w: W - 0.3, h: 0.5, size: 10, valign: "middle" });
  }

  // ---------- 7 お金の話3 何が残る ----------
  {
    const s = add(10);
    heading(s, "お金の話3　うちの財布に、何が残る？", `利用料は${Math.round(V.marginSite() * 100)}%が残る。導入支援は約3割しか残らない。だから利用料を積み上げる`);
    const base = 4.2, sc = 2.35 / 300;
    const bar = (x, v, fill, label, top0 = 0, ink = K.paper) => {
      box(s, { x, y: base - (top0 + v) * sc, w: 0.95, h: v * sc, fill, line: K.ink, lw: 0.75 });
      txt(s, label, { x: x - 0.2, y: base - (top0 + v) * sc - 0.32, w: 1.35, h: 0.3, size: 11, bold: true, align: "center", valign: "bottom", color: K.ink });
    };
    hline(s, X0, base, 5.4, K.ink, 1.5);
    // license
    txt(s, "利用料（毎年）", { x: X0, y: 1.15, w: 3.2, h: 0.28, size: 11, bold: true });
    bar(X0 + 0.1, 300, K.ink, "300万円");
    bar(X0 + 1.2, V.cogsSite(), K.gray500, `−${man(V.cogsSite())}万円`, U.gpYear);
    bar(X0 + 2.3, U.gpYear, ACC, `${man(U.gpYear)}万円`);
    [["受け取る", 0.1], ["AI・保存の費用", 1.2], ["残る（粗利）", 2.3]].forEach(([t, dx]) => txt(s, t, { x: X0 + dx - 0.2, y: base + 0.05, w: 1.35, h: 0.25, size: 9, align: "center" }));
    // setup
    const sx = X0 + 3.5;
    txt(s, "導入支援（初年度だけ）", { x: sx - 0.1, y: 1.15, w: 2.5, h: 0.28, size: 11, bold: true });
    box(s, { x: sx, y: base - 200 * sc, w: 0.7, h: 200 * sc, fill: K.ink, line: K.ink, lw: 0.75 });
    txt(s, "200万円", { x: sx - 0.2, y: base - 200 * sc - 0.32, w: 1.1, h: 0.3, size: 11, bold: true, align: "center", valign: "bottom" });
    box(s, { x: sx + 0.85, y: base - U.setupGp * sc, w: 0.7, h: U.setupGp * sc, fill: K.gray500, line: K.ink, lw: 0.75 });
    txt(s, `${man(U.setupGp)}万円`, { x: sx + 0.65, y: base - U.setupGp * sc - 0.32, w: 1.1, h: 0.3, size: 11, bold: true, align: "center", valign: "bottom" });
    txt(s, "受け取る", { x: sx - 0.2, y: base + 0.05, w: 1.1, h: 0.25, size: 9, align: "center" });
    txt(s, "残る", { x: sx + 0.65, y: base + 0.05, w: 1.1, h: 0.25, size: 9, align: "center" });
    txt(s, `${man(200 - U.setupGp)}万円は\n担当者42人日\nと交通費`, { x: sx + 0.8, y: base - 200 * sc, w: 1.2, h: 0.5, size: 8.5, color: K.gray700 });
    // character
    bubble(s, "人が動く仕事は残りが少ない。\nだから利用料を積み上げる\n会社にします！", { x: 6.15, y: 1.15, w: 2.5, h: 0.95, tail: "br", size: 10 });
    ch(s, "founder-happy", 8.35, 1.6, 1.25);
    box(s, { x: 6.15, y: 2.95, w: 3.45, h: 1.25, fill: K.fill });
    txt(s, [run("粗利（あらり）とは", { bold: true, color: ACC, breakLine: true }), run("売上から、そのサービスを作るのに直接かかったお金を引いた残り。ここから社員の給料や家賃を払う")], { x: 6.3, y: 3.0, w: 3.15, h: 1.15, size: 10, valign: "middle" });
    txt(s, "1拠点10台あたり。AI・保存の費用＝メガネ1台 月約6,000円＋会社ごとのデータ置き場 月1万円（大事な場面だけ送る前提）", { x: X0, y: 4.75, w: W, h: 0.26, size: S.note, color: K.gray500 });
  }

  // ---------- 8 お金の話4 3拠点 ----------
  {
    const s = add(11);
    heading(s, "お金の話4　なぜ赤字が続く？ 合言葉は「4」", `社員1人で約4拠点を見られれば黒字。10台のままでは5年目でも${Y5.sitesPerStaff.toFixed(2)}`);
    const top = 1.2;
    box(s, { x: X0, y: top, w: 4.3, h: 1.55, fill: K.fill });
    txt(s, [run("社員1人にかかる費用", { breakLine: true }), run(`年${man(PER)}万円`, { bold: true, fontSize: 20 })], { x: X0 + 0.1, y: top + 0.1, w: 2.0, h: 0.8, size: 9.5, align: "center" });
    txt(s, "÷", { x: X0 + 2.0, y: top + 0.25, w: 0.3, h: 0.5, size: 22, bold: true, align: "center" });
    txt(s, [run("1拠点から残る粗利", { breakLine: true }), run(`年${man(U.gpYear)}万円`, { bold: true, fontSize: 20 })], { x: X0 + 2.25, y: top + 0.1, w: 2.0, h: 0.8, size: 9.5, align: "center" });
    txt(s, [run(`＝ ${(PER / U.gpYear).toFixed(1)}　→　`), run("目安 約4拠点", { color: ACC, fontSize: 24 })], { x: X0, y: top + 0.82, w: 4.3, h: 0.45, size: 16, bold: true, align: "center", valign: "middle" });
    txt(s, "新しいお客様は最初の年は半年分しか払わないので、年度末に数えると約4", { x: X0 + 0.1, y: top + 1.25, w: 4.1, h: 0.26, size: 8.5, color: K.gray700, align: "center" });
    ch(s, "founder-normal", X0 + 0.1, top + 1.75, 1.3);
    [0, 1, 2, 3].forEach((i) => ic(s, "office", X0 + 1.35 + i * 0.68, top + 2.15, 0.6));
    txt(s, "1人で、約4拠点（支店・営業所）を見る", { x: X0 + 1.3, y: top + 2.85, w: 2.9, h: 0.28, size: 10, bold: true });
    const gx = 5.05, gw = 4.55, base = 4.15, sc = 2.3 / 4.5;
    txt(s, "社員1人あたりの拠点数", { x: gx, y: top, w: gw, h: 0.26, size: 10.5, bold: true });
    hline(s, gx, base, gw, K.ink, 1.5);
    R.forEach((r, i) => {
      const v = r.sitesPerStaff, x = gx + 0.2 + i * 0.86, black = r.op > 0;
      box(s, { x, y: base - v * sc, w: 0.55, h: v * sc, fill: black ? ACC : K.ink });
      txt(s, v.toFixed(2), { x: x - 0.15, y: base - v * sc - 0.26, w: 0.85, h: 0.24, size: 10, bold: true, align: "center" });
      txt(s, r.year + "年目", { x: x - 0.15, y: base + 0.04, w: 0.85, h: 0.22, size: 9, align: "center" });
      txt(s, `${r.sites}拠点÷${r.staff}人`, { x: x - 0.2, y: base + 0.25, w: 0.95, h: 0.2, size: 7.5, align: "center", color: K.gray700 });
    });
    s.addShape("line", { x: gx, y: base - 4 * sc, w: gw, h: 0, line: { color: ACC, width: 1.75, dashType: "dash" } });
    txt(s, "目安 4", { x: gx + 0.05, y: base - 4 * sc - 0.27, w: 1.3, h: 0.24, size: 9.5, bold: true, color: ACC });
    bubble(s, "人を先に雇うから\n5年目でもまだ赤字。\n台数を増やして届かせる！", { x: gx + 0.05, y: top + 0.25, w: 2.0, h: 0.8, tail: "bl", size: 9.5 });
    txt(s, `社員1人の費用＝5年目の販管費${man(Y5.sga)}万円÷${Y5.staff}人（営業・技術・導入コンサルタントの全員。給料・会社負担・採用費・家賃などを含む）`, { x: X0, y: 4.75, w: W, h: 0.26, size: S.note, color: K.gray500 });
  }

  // ---------- 9 お金の話5 雪だるま ----------
  {
    const s = add(12);
    heading(s, "お金の話5　去年のお客様が今年も払う。売上は雪だるま式", `5年目の売上約${man(Y5.sales / 100)}百万円のうち${Math.round(Y5.existing / Y5.sales * 100)}%は、前の年までのお客様の利用料（解約は年1割で引く）`);
    const years = R.map((r) => r.year + "年目");
    const ser = [
      ["前の年までのお客様の利用料", R.map((r) => r.existing / 100), K.ink],
      ["新しいお客様の利用料", R.map((r) => r.newLic / 100), ACC],
      ["導入支援", R.map((r) => r.setup / 100), K.gray500],
      ["個社設定", R.map((r) => r.contract / 100), K.rule],
    ];
    s.addChart(pres.charts.BAR, ser.map(([name, values]) => ({ name, labels: years, values })), {
      x: X0, y: 1.15, w: 5.6, h: 3.5, barDir: "col", barGrouping: "stacked", chartColors: ser.map((x) => x[2]), barGapWidthPct: 60,
      catAxisLabelColor: K.gray700, valAxisLabelColor: K.gray700, catAxisLabelFontSize: 9, valAxisLabelFontSize: 8, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
      valGridLine: { color: K.rule, size: 0.5 }, valAxisMaxVal: 300, valAxisMajorUnit: 50, valAxisLineShow: false,
      showLegend: true, legendPos: "t", legendFontSize: 8, legendFontFace: "+mn-lt", objectName: "snowball",
    });
    const rx = 6.25, rw = 3.35;
    txt(s, "売上（百万円）", { x: rx, y: 1.15, w: rw, h: 0.26, size: 10.5, bold: true });
    L.table(s, [["", "売上", "営業利益"], ...R.map((r) => [r.year + "年目", String(M(r.sales)), { text: L.tri(M(r.op)), color: r.op < 0 ? L.RED : L.GREEN, bold: true }])],
      { x: rx, y: 1.45, w: rw, colW: [1.15, 1.1, 1.1], rowH: 0.27, size: 10, name: "sales-op" });
    txt(s, [run("10年なら：", { bold: true, color: L.NAVY }), run(`${R10.firstBlack}年目に黒字、${R10.cumBack}年目に赤字を回収`)], { x: rx, y: 3.12, w: rw, h: 0.26, size: 9.5, valign: "middle", fill: L.PALEY });
    ch(s, "founder-happy", rx + 2.05, 3.45, 1.3);
    bubble(s, "新しいお客様が\n増えるたびに\n大きくなる！", { x: rx, y: 3.5, w: 1.95, h: 0.95, tail: "br", size: 10 });
    txt(s, "新しいお客様は年の途中で始めるので、初年度の利用料は半年分で計算。1年目の3拠点は試験導入（利用料なし）", { x: X0, y: 4.75, w: W, h: 0.26, size: S.note, color: K.gray500 });
  }

  // ---------- 10 お金の話6 LTV/CAC ----------
  {
    const s = add(13);
    heading(s, `お金の話6　${man(U.cac)}万円のエサで、${man(U.ltv(7))}万円の魚を釣る`, "1拠点を連れてくる費用（CAC）と、1拠点が残してくれる粗利（LTV）");
    const top = 1.2;
    box(s, { x: X0, y: top, w: 3.6, h: 2.85, line: K.ink, lw: 1.25 });
    ic(s, "bait", X0 + 0.15, top + 0.2, 1.1);
    txt(s, [run("連れてくる費用", { breakLine: true }), run("CAC（キャック）", { fontSize: 9, color: K.gray700 })], { x: X0 + 1.3, y: top + 0.25, w: 2.2, h: 0.55, size: 12, bold: true });
    txt(s, `約${man(U.cac)}万円`, { x: X0 + 1.3, y: top + 0.8, w: 2.2, h: 0.5, size: 24, bold: true });
    txt(s, [run(`（営業3人の人件費${man(3 * U.salesPay)}万円`, { breakLine: true }), run("＋広告・展示会500万円）", { breakLine: true }), run("÷ 新しいお客様33拠点（5年目）")], { x: X0 + 0.15, y: top + 1.55, w: 3.3, h: 0.9, size: 10 });
    txt(s, `約${Math.round(U.ltv(7) / U.cac)}倍`, { x: 4.05, y: top + 0.85, w: 1.5, h: 0.55, size: 24, bold: true, color: ACC, align: "center" });
    arrow(s, 4.1, top + 1.5, 1.4, 0, { lw: 3, color: ACC });
    box(s, { x: 5.6, y: top, w: 4.0, h: 2.85, fill: K.ink });
    ic(s, "fish", 5.75, top + 0.2, 1.1);
    txt(s, [run("残してくれる粗利", { breakLine: true }), run("LTV（エルティーブイ）", { fontSize: 9 })], { x: 6.95, y: top + 0.25, w: 2.55, h: 0.55, size: 12, bold: true, color: K.paper });
    txt(s, `約${man(U.ltv(7))}万円`, { x: 6.95, y: top + 0.8, w: 2.6, h: 0.5, size: 24, bold: true, color: L.YELLOW });
    txt(s, [run(`年${man(U.gpYear)}万円 × 7年`, { breakLine: true }), run(`＋ 導入支援の粗利${man(U.setupGp)}万円`, { breakLine: true }), run("7年は控えめ。解約が年1割なら平均10年続く", { fontSize: 9 })], { x: 5.75, y: top + 1.55, w: 3.7, h: 0.95, size: 10.5, color: K.paper });
    box(s, { x: X0, y: 4.25, w: W, h: 0.8, fill: K.fill });
    ch(s, "client-happy", X0 + 0.1, 4.28, 0.75);
    txt(s, [run(`${man(U.cac)}万円は約${Math.round(U.paybackMonths)}か月で回収（${man(U.cac)}万円 ÷ 月${man(U.gpYear / 12)}万円の粗利）。`, { bold: true, breakLine: true }), run(`目安は「LTVがCACの3倍以上」。当社は約${Math.round(U.ltv(7) / U.cac)}倍なので、お客様を増やすほど会社が強くなる`)], { x: X0 + 0.85, y: 4.25, w: W - 1.0, h: 0.8, size: 10.5, valign: "middle" });
  }

  // ---------- 10-2 お金の話7 うまくいかなかったら ----------
  {
    const s = add(14);
    heading(s, "お金の話7　どうすれば黒字？ 条件は1拠点の台数", "10台のままだと計画どおりでも5年で黒字にならない。15台に増やせれば6割でも5年目に黒字");
    L.gates(s, V);
    txt(s, "上の表：新しい拠点が計画の何割だったら、何年目に黒字になるか（百万円）。下の表：進む・直す・止まるを先に決めておく", { x: X0, y: 4.98, w: W, h: 0.24, size: S.note, color: K.gray500 });
  }

  // ---------- 11 500万円 ----------
  {
    const s = add(15);
    heading(s, "準備期の6か月で、最初の1拠点を動かす", "費用は約500万円。自己資金でまかない、創業者の給料はゼロ");
    const top = 1.2;
    const steps = [["はじまり", "準備期", "自己資金で約500万円"], ["3か月", "試作品ができる", "記録→写真抽出→報告書"], ["6か月", "1拠点で実証", `1台1日${Math.round(CV.breakEvenMin)}分減ったかを測る`], ["その後", "自己資金2億円を入れる", "実証に合格したら採用を始める"]];
    steps.forEach(([t, a, b], i) => {
      const x = X0 + i * 2.33;
      box(s, { x, y: top, w: 2.15, h: 1.5, fill: i === 0 ? K.ink : K.fill, line: i === 0 ? undefined : K.rule });
      const c = i === 0 ? K.paper : K.ink;
      txt(s, t, { x: x + 0.12, y: top + 0.1, w: 1.9, h: 0.3, size: 11, bold: true, color: i === 0 ? L.YELLOW : ACC });
      txt(s, a, { x: x + 0.12, y: top + 0.45, w: 1.95, h: 0.5, size: 12, bold: true, color: c });
      txt(s, b, { x: x + 0.12, y: top + 0.98, w: 1.95, h: 0.45, size: 9, color: c });
      if (i < 3) arrow(s, x + 2.17, top + 0.75, 0.15, 0, { lw: 2 });
    });
    const uses = [["試作品の開発", 250], ["実証（1拠点・3か月）", 100], ["メガネ・機材", 80], ["設立・契約", 40], ["予備", 30]];
    txt(s, "準備期の費用（万円）", { x: X0, y: 2.95, w: 4, h: 0.26, size: 10.5, bold: true });
    const k = 2.4 / 250;
    uses.forEach(([n, v], i) => {
      const y = 3.27 + i * 0.33;
      txt(s, n, { x: X0, y, w: 1.75, h: 0.28, size: 9.5, valign: "middle" });
      box(s, { x: X0 + 1.8, y: y + 0.04, w: v * k, h: 0.2, fill: i === 0 ? ACC : K.ink });
      txt(s, String(v), { x: X0 + 1.85 + v * k, y, w: 0.5, h: 0.28, size: 10, bold: true, valign: "middle" });
    });
    box(s, { x: 5.0, y: 2.95, w: 4.6, h: 1.95, line: ACC, lw: 2 });
    ch(s, "founder-normal", 5.15, 3.2, 1.4);
    txt(s, [run("実証の合格ライン", { bold: true, color: ACC, fontSize: 11, breakLine: true }), run(`1台1日${Math.round(CV.breakEvenMin)}分以上減れば進む`, { bold: true, fontSize: 13, breakLine: true }), run("15分未満なら撤退。その間なら機能を写真台帳に絞る", { fontSize: 9, breakLine: true }), run("お願い：実証先の電気工事会社のご紹介", { fontSize: 9.5 })], { x: 6.4, y: 3.05, w: 3.1, h: 1.75, size: 10, valign: "middle", psa: 2 });
  }

  // ---------- 12 全体図 ----------
  {
    const s = add(16);
    heading(s, "1枚で言うと ─ 上は記録とサービス、下はお金", "記録が貯まるほど他社がすぐに真似できなくなり、お客様が増えるほど黒字が厚くなる");
    const top = 1.15;
    // customer left, us center, costs right
    box(s, { x: X0, y: top + 0.35, w: 2.1, h: 3.2, line: K.ink, lw: 1.5 });
    ic(s, "office", X0 + 0.5, top + 0.5, 1.1);
    txt(s, "電気工事会社（お客様）", { x: X0, y: top + 1.75, w: 2.1, h: 0.28, size: 11, bold: true, align: "center" });
    ch(s, "veteran-normal", X0 + 0.15, top + 2.15, 0.95);
    ch(s, "junior-happy", X0 + 1.0, top + 2.15, 0.95);
    txt(s, "熟練者・若手・工事部長", { x: X0, y: top + 3.15, w: 2.1, h: 0.25, size: 8.5, align: "center", color: K.gray700 });

    const cx = 3.75, cw = 2.45;
    box(s, { x: cx, y: top + 0.35, w: cw, h: 3.2, fill: K.ink });
    txt(s, "カンツギ", { x: cx, y: top + 0.45, w: cw, h: 0.3, size: 12, bold: true, color: K.paper, align: "center" });
    ic(s, "server", cx + 0.95, top + 0.85, 0.7);
    txt(s, [run("AIが整理", { bold: true, breakLine: true }), run("記録が貯まるほど、他社が", { breakLine: true }), run("すぐに真似できなくなる", { color: L.YELLOW, bold: true })], { x: cx + 0.1, y: top + 1.6, w: cw - 0.2, h: 0.85, size: 10, color: K.paper, align: "center" });
    txt(s, [run(`1拠点の粗利 ${man(U.gpYear)}万円`, { bold: true, breakLine: true }), run(`（300万円 − AI・保存${man(V.cogsSite())}万円）`)], { x: cx + 0.1, y: top + 2.55, w: cw - 0.2, h: 0.6, size: 9.5, color: K.paper, align: "center" });

    // top arrows: service
    arrow(s, 2.6, top + 0.75, 1.05, 0, { lw: 2.5 });
    txt(s, "① 動画と会話", { x: 2.5, y: top + 0.42, w: 1.25, h: 0.28, size: 9, bold: true, align: "center" });
    arrow(s, 2.6, top + 1.6, 1.05, 0, { lw: 2.5, flipH: true });
    txt(s, "② 照合・報告書\n・教材", { x: 2.5, y: top + 1.05, w: 1.25, h: 0.5, size: 9, bold: true, align: "center" });
    // bottom arrow: money
    arrow(s, 2.6, top + 2.95, 1.05, 0, { lw: 4, color: ACC });
    ic(s, "coin", 2.9, top + 2.3, 0.5);
    txt(s, "③ 年300万円（10台）", { x: 2.5, y: top + 3.05, w: 1.25, h: 0.28, size: 10, bold: true, color: ACC, align: "center" });

    // right: costs
    const kx = 6.65, kw = 2.95;
    arrow(s, cx + cw + 0.04, top + 1.05, 0.38, 0, { lw: 2.5 });
    arrow(s, cx + cw + 0.04, top + 2.75, 0.38, 0, { lw: 2.5 });
    box(s, { x: kx, y: top + 0.35, w: kw, h: 1.4, fill: K.fill });
    txt(s, [run("④ 出ていくお金", { bold: true, color: ACC, breakLine: true }), run("AI・保存：メガネ1台 月約6,000円", { breakLine: true }), run(`社員1人：年${man(PER)}万円（給料・家賃など）`)], { x: kx + 0.12, y: top + 0.4, w: kw - 0.24, h: 1.3, size: 10, valign: "middle", psa: 3 });
    box(s, { x: kx, y: top + 2.0, w: kw, h: 1.55, line: ACC, lw: 2 });
    txt(s, [run("⑤ 黒字の条件", { bold: true, color: ACC, breakLine: true }), run("社員1人 × 約4拠点", { bold: true, fontSize: 15, breakLine: true }), run(`10台のままでは5年目でも${Y5.sitesPerStaff.toFixed(2)}`, { breakLine: true }), run(`15台に増やせれば${F15}年目に黒字`)], { x: kx + 0.12, y: top + 2.05, w: kw - 0.24, h: 1.45, size: 10, valign: "middle", psa: 2 });
    txt(s, "将来：貯まった作業の記録で、ロボットに作業を教える（フィジカルAI）", { x: X0, y: 4.85, w: W, h: 0.26, size: 9.5, bold: true, color: K.gray700 });
  }

  // ---------- 13 ツッコミ対策 ----------
  {
    const s = add(17);
    heading(s, "よくあるツッコミ4つ ─ この一言で返す", "答えは必ず数字から言う。理由は後");
    const qa = [
      ["本当に年300万円も払うの？", `1台1日${Math.round(CV.breakEvenMin)}分減れば元が取れます。想定は35分で、実証で測ります`, `${man(CV.cost)}万円÷（10台×240日）÷1時間の人件費`],
      ["なんで赤字が続くの？", "人を先に雇うからです。社員1人約4拠点で黒字。台数を増やして早めます", `${man(PER)}万円÷${man(U.gpYear)}万円＝${(PER / U.gpYear).toFixed(1)}、半年分を見て約4`],
      ["真似されたら終わりでは？", "現場の記録は使うほど貯まり、後から来た会社には移せません", "貯まる・意味づけ・業界の型"],
      ["日立と同じでは？", "日立は保守契約の中の安全・点検。当社は中堅の電気工事会社の書類と育成を月額で", "公表資料で比べた（本編 p.15）"],
    ];
    qa.forEach(([q, a, why], i) => {
      const y = 1.12 + i * 0.94;
      ch(s, "client-think", X0, y, 0.85);
      bubble(s, q, { x: X0 + 0.8, y: y + 0.06, w: 2.75, h: 0.52, tail: "bl", size: 10, bold: true });
      bubble(s, a, { x: 3.95, y: y + 0.02, w: 4.6, h: 0.58, tail: "br", size: 9.5, fill: K.fill });
      ch(s, "founder-happy", 8.7, y, 0.85);
      txt(s, "根拠：" + why, { x: 4.0, y: y + 0.69, w: 4.5, h: 0.2, size: 8, color: ACC, bold: true });
    });
    txt(s, "左：聞き手　右：社長（わたし）。細かい計算は「計算詳細版」に", { x: X0, y: 4.92, w: W, h: 0.24, size: S.note, color: K.gray500 });
  }

  // ---------- 14 まとめ ----------
  {
    const s = add(18);
    heading(s, "まとめ ─ この3つの数字だけ覚えれば、話せる", `300万・${man(U.gpYear)}万・4。順番に言えば、お金の仕組みが全部つながる`);
    const top = 1.15;
    const big = [["300万円", "お客様が10台で毎年払う", `1台1日${Math.round(CV.breakEvenMin)}分減れば元が取れる`], [`${man(U.gpYear)}万円`, "そのうち当社に残る粗利", `AI・保存の${man(V.cogsSite())}万円を引いた残り`], ["約4拠点", "社員1人で見れば黒字", `10台のままでは5年目${Y5.sitesPerStaff.toFixed(2)}、15台なら${F15}年目に黒字`]];
    big.forEach(([n, a, b], i) => {
      const x = X0 + i * 3.1;
      box(s, { x, y: top, w: 2.95, h: 1.55, fill: i === 2 ? K.ink : K.fill });
      const c = i === 2 ? K.paper : K.ink;
      txt(s, n, { x: x + 0.15, y: top + 0.1, w: 2.7, h: 0.65, size: 30, bold: true, color: i === 2 ? L.YELLOW : ACC, valign: "middle" });
      txt(s, a, { x: x + 0.15, y: top + 0.78, w: 2.7, h: 0.3, size: 11.5, bold: true, color: c });
      txt(s, b, { x: x + 0.15, y: top + 1.1, w: 2.7, h: 0.3, size: 9.5, color: c });
      if (i < 2) arrow(s, x + 2.97, top + 0.78, 0.11, 0, { lw: 2 });
    });
    box(s, { x: X0, y: 2.9, w: W, h: 1.95, line: K.ink, lw: 1.25, name: "script30" });
    txt(s, "30秒で言うと（そのまま読める台本）", { x: X0 + 0.15, y: 2.97, w: 5, h: 0.28, size: 10.5, bold: true, color: ACC });
    txt(s, `電気工事会社が、メガネ10台で年300万円を払います。掛けた班の書類の時間が1台1日${Math.round(CV.breakEvenMin)}分減れば元が取れ、想定の35分なら約${CV.ratio.toFixed(1)}倍の効果です。当社にはAIと保存の費用を引いた${man(U.gpYear)}万円が残り、社員1人で約4拠点を見れば黒字です。人を先に雇うので10台のままでは5年目も赤字ですが、効果が出た班から15台に増やせば${F15}年目に黒字になります。10年で見ると、10台のままでも${R10.firstBlack}年目に黒字です。`, { x: X0 + 0.15, y: 3.26, w: W - 1.6, h: 1.55, size: 10.5, lsm: 1.25 });
    ch(s, "founder-happy", 8.35, 3.3, 1.35);
    txt(s, "ご清聴ありがとうございました", { x: 3.2, y: 4.88, w: 3.3, h: 0.28, size: 11, bold: true, color: L.NAVY, align: "right", valign: "middle" });
  }

  fs.mkdirSync(OUT_DIR, { recursive: true });
  await pres.writeFile({ fileName: OUT });
  console.log("wrote", OUT);
}

main().catch((e) => { console.error(e); process.exit(1); });
