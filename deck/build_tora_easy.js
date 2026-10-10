// 令和の虎 事業計画書「やさしい版（マンガ）」: story and the money mechanics in comic panels.
// Usage: node build_tora_easy.js [outDir]   (needs pptxgenjs and sharp)
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const L = require("./tora_lib.js");
const { run: model } = require("./tora_model.js");

const { K, S, X0, W, ACC, txt, box, hline, arrow, run, heading, panel, bubble, sfx, png, charSVG, iconSVG } = L;
const OUT_DIR = process.argv[2] || __dirname;
const OUT = path.join(OUT_DIR, "令和の虎_やさしい版_マンガでわかる収益構造.pptx");
const R = model().rows;
const M = (v) => Math.round(v / 100); // 万円 → 百万円

// ---------- notes: what to say, in spoken words ----------
const NOTE_PARTS = {
  1: ["表紙・20秒", "数字は3つだけ。300万円、255万円、2.8拠点",
    "三現ワークスです。今日はお金の仕組みを、3つの数字だけでお話しします。お客様が1拠点で年300万円払う。そのうち255万円が当社に残る。社員1人で2.8拠点を見られれば黒字。この3つです。"],
  2: ["登場人物・30秒", "お金を払うのは電気工事会社。窓口は工事部長",
    "登場人物は4人です。来年定年の熟練の電工、田中さん。入社2年目の施工管理、佐藤くん。報告書と育成で手一杯の山本工事部長。そして私です。ここで大事なのは、お金を払うのは現場の人ではなく会社、つまり工事部長や社長だということです。"],
  3: ["第1話・40秒", "熟練者の勘は書かれていない。辞めたら止まる",
    "ある日の工事現場。ケーブルの接続で、田中さんは、端末の寸法はテープの巻き始めから測れと言う。佐藤くんには理由が分からない。手順書にも書いていない。工事部長は、来年田中さんが辞めたらどうなるのかと青ざめる。しかも2024年から残業に上限がかかり、工事写真と報告書で教える時間もない。建設業で働く人の3人に1人以上が55歳以上です。これが、全国の工事会社で起きていることです。"],
  4: ["第2話・40秒", "メガネを掛けて、いつも通り働くだけ",
    "そこで、スマートグラスを掛けてもらいます。田中さんは、いつも通り作業するだけ。動画と会話をAIが整理して、三つの形で渡します。若手には、図面とのズレをその場で表示。管理者には、報告書を自動で作成。そして後から、教材にします。"],
  5: ["第3話・40秒", "会う → 試す → 決める → 広がる",
    "最初のお客様は、関東の中堅の電気工事会社です。業界の勉強会で、2024年からの残業の上限と書類の削減について話します。展示会では、デモを自分の現場に当てはめて試してもらいます。次に、3か月の有償の試験導入で、報告書の時間が本当に減るかを数字で確かめます。本契約になったら、協力会社にも同じ形で使ってもらいます。関東の中堅735社のうち、半分近くに会い、4社に1社と契約できれば、5年で約83社です。"],
  6: ["お金の話1・40秒", "1拠点 年300万円。初年度だけ導入支援200万円",
    "では、誰が何にいくら払うのか。払うのは電気工事会社です。1拠点、つまり1つの支店や営業所で、年300万円。月にすると25万円です。中身は、アプリとデータ置き場の基本料が100万円、記録する工区や現場の数で100万円、教材づくりなどの追加機能で100万円。初年度だけ、私たちが2か月現場に入る導入支援に200万円いただきます。スマホの契約と同じで、初期設定は最初だけ、月額は毎月です。"],
  7: ["お金の話2・40秒", "月25万円払って、月240万円浮く。約2か月で元が取れる",
    "なぜ工事部長は払うのか。施工管理者と電工が100人の会社で、1人あたり日報や写真整理が月6時間減ると、時給4,000円換算で月240万円分の時間が浮きます。初年度の支払いは500万円なので、約2か月で元が取れます。月6時間とは、1日あたり約18分。朝の日報と夕方の写真整理がなくなるイメージです。"],
  8: ["お金の話3・40秒", "利用料300万円のうち255万円が残る。だから利用料を積み上げる",
    "次に、当社の財布に何が残るか。300万円の利用料のうち、サーバーやAIの費用は15%の45万円。255万円が残ります。導入支援の200万円は、人が動くので120万円が人件費に消え、残るのは80万円。だから当社は、人が動く仕事ではなく、利用料を積み上げる会社にします。"],
  9: ["お金の話4・40秒", "社員1人で2.8拠点を見れば黒字。4年目に超える",
    "なぜ最初は赤字なのか。社員1人にかかる費用は、給料や家賃を合わせて年713万円。1拠点から残るのは255万円。713を255で割ると2.8。つまり、社員1人で2.8拠点を見られれば黒字です。1年目から3年目は、人を先に雇うので2.8に届かず赤字。4年目に3.67となって黒字になります。合言葉は2.8です。"],
  10: ["お金の話5・40秒", "去年のお客様が今年も払う。売上は雪だるま式",
    "利用料は毎年いただくので、去年までのお客様の分が今年の売上に乗ります。5年目の売上2億9,000万円のうち、57%は前の年までのお客様からの利用料です。新しいお客様が増えるたびに、雪だるまのように大きくなります。営業利益は4年目に黒字、5年目に6,900万円です。"],
  11: ["お金の話6・30秒", "64万円かけて連れてきたお客様が、1,865万円残してくれる",
    "最後に、お客様1拠点を連れてくる費用は約64万円。営業2人の人件費と広告費を、新しいお客様の数で割った数字です。一方、1拠点が7年間で残してくれる粗利は約1,865万円。約29倍です。64万円は3か月で回収できます。7年は控えめに見た数字です。"],
  12: ["500万円・30秒", "500万円で、6か月以内に1拠点を動かす",
    "今回お願いする500万円は、試作品と1拠点の実証に使います。私の給料はゼロです。3か月で試作品、6か月で実証。その結果を持って、公庫の融資とベンチャーキャピタルの出資を受けます。条件は10%でのご提案です。"],
  13: ["全体図・30秒", "上の矢印は記録とサービス、下の矢印はお金",
    "1枚にまとめるとこうなります。上の流れが記録とサービス。お客様の現場から記録が来て、照合・報告書・教材にして返す。下の流れがお金。お客様から年300万円が入り、45万円がサーバー代、255万円が残って社員の費用をまかなう。社員1人で2.8拠点を超えたら黒字。そして記録が貯まるほど、真似されにくくなります。"],
  14: ["ツッコミ対策・40秒", "3つのツッコミには、この一言で返す",
    "よく聞かれる3つの質問への答えです。本当に300万円も払うのか。月240万円分の時間が浮くので、2か月で元が取れます。なぜ4年目まで赤字なのか。人を先に雇うからで、社員1人2.8拠点を超えたら黒字になります。真似されないか。現場の記録は使うほど貯まり、後から来た会社には移せません。"],
  15: ["まとめ・30秒", "300万・255万・2.8。この3つで話せる",
    "まとめです。お客様が年300万円払う。255万円が残る。社員1人2.8拠点で黒字。この3つの数字で、すべて説明できます。詳しい計算の根拠は、別冊の理論武装版にまとめています。"],
};
const NOTES = Object.fromEntries(Object.entries(NOTE_PARTS).map(([n, [t, msg, body]]) => [n, `【${t}】\n要約：${msg}\n\n${body}`]));

async function main() {
  const pres = new pptxgen();
  const add = await L.setup(pres, "マンガでわかる 三現ワークスのもうけの仕組み", NOTES);

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
    txt(s, "令和の虎　事業計画書　やさしい版", { x: X0, y: 0.35, w: 5.5, h: 0.3, size: 11, color: K.gray700 });
    txt(s, [run("マンガでわかる", { breakLine: true }), run("三現ワークスの"), run("もうけの仕組み", { color: ACC })], { x: X0, y: 0.68, w: 6.0, h: 1.1, size: 28, bold: true, lsm: 1.1 });
    box(s, { x: 6.75, y: 0.45, w: 2.85, h: 1.25, line: ACC, lw: 2, name: "ask" });
    txt(s, "希望金額", { x: 6.95, y: 0.52, w: 2, h: 0.28, size: 11, bold: true, color: K.gray700 });
    txt(s, "500万円", { x: 6.95, y: 0.8, w: 2.6, h: 0.8, size: 38, bold: true, color: ACC, valign: "middle" });
    const nums = [["お客様が払う", "300万円", "1拠点・1年あたり"], ["当社に残る", "255万円", "サーバー代を\n引いた粗利"], ["黒字のライン", "2.8拠点", "社員1人あたり"]];
    nums.forEach(([a, b, c], i) => {
      const x = X0 + i * 3.1, y = 2.0;
      box(s, { x, y, w: 2.95, h: 1.0, fill: i === 1 ? K.ink : K.fill, name: "num-" + i });
      const col = i === 1 ? K.paper : K.ink;
      txt(s, [run(String(i + 1) + "  ", { color: ACC, bold: true }), run(a)], { x: x + 0.15, y: y + 0.08, w: 2.7, h: 0.28, size: 11, bold: true, color: col });
      txt(s, b, { x: x + 0.15, y: y + 0.36, w: 1.55, h: 0.55, size: 22, bold: true, color: i === 1 ? "E3A774" : ACC, valign: "middle" });
      txt(s, c, { x: x + 1.65, y: y + 0.4, w: 1.25, h: 0.5, size: 8, color: col, valign: "middle" });
    });
    const people = [["veteran-normal", "熟練者"], ["junior-normal", "若手"], ["manager-worried", "工事部長"], ["founder-happy", "社長（わたし）"]];
    people.forEach(([k, n], i) => {
      const x = X0 + 0.1 + i * 1.25;
      ch(s, k, x, 3.25, 1.5);
      txt(s, n, { x: x - 0.1, y: 4.78, w: 1.45, h: 0.24, size: 9, align: "center", color: K.gray700 });
    });
    bubble(s, "数字は3つだけ。\nこれで全部、説明できます！", { x: 5.6, y: 3.45, w: 3.4, h: 0.95, tail: "bl", size: 13, bold: true, name: "cover-bubble" });
  }

  // ---------- 2 登場人物 ----------
  {
    const s = add(2);
    heading(s, "登場人物 ─ お金を払うのは「会社」。窓口は工事部長", "この4人で、困りごと → 解決 → お金の流れを追いかける");
    const cast = [
      ["veteran-normal", "田中さん（62歳）", "熟練の電工。来年、定年", "勘で分かる。\nでも説明はできん"],
      ["junior-worried", "佐藤くん（24歳）", "入社2年目の施工管理", "田中さんが辞めたら\nどうしよう…"],
      ["manager-worried", "山本工事部長（50歳）", "管理者。お金を払う人", "報告書と育成で\n手一杯だ"],
      ["founder-happy", "社長（わたし）", "三現ワークス。元・施工管理", "現場の勘を、\n次の担い手へ"],
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
    txt(s, [run("ポイント：", { bold: true, color: ACC }), run("使うのは現場、払うのは会社。だから「会社にとって得か」で値段を決める")], { x: X0 + 0.15, y: 4.72, w: W - 0.3, h: 0.38, size: 10.5, valign: "middle" });
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
    txt(s, [run("現実も同じ：", { bold: true, color: "E3A774" }), run("建設業で働く人の36.7%が55歳以上、29歳以下は11.7%（国交省、2024年）。2024年4月から残業にも上限")], { x: X0 + 0.15, y: 4.72, w: W - 0.3, h: 0.38, size: 9, color: K.paper, valign: "middle" });
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

  // ---------- 5 第3話 どう会う ----------
  {
    const s = add(5);
    heading(s, "第3話　最初のお客様に、どう会う？", "関東の中堅電気工事会社（施工管理者30人以上）から。会う → 試す → 決める → 広がる");
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
    txt(s, [run("5年の見込み：", { bold: true, color: "E3A774" }), run("関東の中堅735社 → 会う45% → 契約25% ＝ 約83社。大手1社を目玉の事例にして協力会社へ広げる（率は仮定）")], { x: X0 + 0.15, y: 4.72, w: W - 0.3, h: 0.38, size: 9.5, color: K.paper, valign: "middle" });
  }

  // ---------- 5 お金の話1 誰が何にいくら ----------
  {
    const s = add(6);
    heading(s, "お金の話1　誰が、何に、いくら払う？", "払うのは電気工事会社（1拠点＝支店・営業所）。毎年300万円、初年度だけ導入支援200万円");
    const top = 1.2;
    // flow: customer -> coin -> us
    ic(s, "office", X0 + 0.15, top + 0.3, 1.1);
    ch(s, "manager-normal", X0 + 0.25, top + 1.6, 1.15);
    txt(s, "電気工事会社（お客様）", { x: X0 - 0.25, y: top + 2.8, w: 1.95, h: 0.26, size: 10, bold: true, align: "center" });
    ch(s, "founder-happy", X0 + 3.6, top + 1.0, 1.55);
    txt(s, "三現ワークス", { x: X0 + 3.35, y: top + 2.6, w: 1.8, h: 0.26, size: 10, bold: true, align: "center" });
    arrow(s, X0 + 1.55, top + 1.0, 1.9, 0, { lw: 4, color: ACC });
    ic(s, "coin", X0 + 2.15, top + 0.25, 0.6);
    txt(s, [run("年300万円", { bold: true, fontSize: 16, color: ACC, breakLine: true }), run("（月25万円）", { fontSize: 10 })], { x: X0 + 1.5, y: top + 1.12, w: 2.0, h: 0.6, align: "center" });
    arrow(s, X0 + 1.55, top + 2.1, 1.9, 0, { lw: 2, color: K.gray500, dash: "dash" });
    txt(s, "初年度だけ ＋200万円\n（導入支援）", { x: X0 + 1.5, y: top + 2.18, w: 2.0, h: 0.5, size: 9.5, align: "center", color: K.gray700 });
    // receipt
    const rx = 5.55, rw = 4.05;
    box(s, { x: rx, y: top - 0.05, w: rw, h: 3.0, line: K.ink, lw: 1.25, name: "receipt" });
    txt(s, "御請求書（1拠点・1年分）", { x: rx + 0.2, y: top + 0.05, w: rw - 0.4, h: 0.32, size: 12, bold: true, align: "center" });
    hline(s, rx + 0.2, top + 0.42, rw - 0.4, K.ink, 1.5);
    const lines = [["基本料", "アプリとデータ置き場", "100万円"], ["現場登録", "記録する工区・現場 20万円×5", "100万円"], ["追加機能", "教材づくり等 50万円×2つ", "100万円"]];
    lines.forEach(([a, b, c], i) => {
      const y = top + 0.5 + i * 0.42;
      txt(s, [run(a, { bold: true, breakLine: true }), run(b, { fontSize: 8.5, color: K.gray700 })], { x: rx + 0.2, y, w: 2.6, h: 0.4, size: 10.5 });
      txt(s, c, { x: rx + 2.8, y, w: 1.05, h: 0.4, size: 11, align: "right", valign: "middle" });
    });
    hline(s, rx + 0.2, top + 1.8, rw - 0.4, K.ink, 1.5);
    txt(s, "合計（毎年）", { x: rx + 0.2, y: top + 1.86, w: 2, h: 0.4, size: 12, bold: true, valign: "middle" });
    txt(s, "300万円", { x: rx + 2.2, y: top + 1.86, w: 1.65, h: 0.4, size: 18, bold: true, color: ACC, align: "right", valign: "middle" });
    s.addShape("line", { x: rx + 0.2, y: top + 2.35, w: rw - 0.4, h: 0, line: { color: K.gray500, width: 1, dashType: "dash" } });
    txt(s, [run("初年度だけ　導入支援 200万円", { bold: true, breakLine: true }), run("当社の担当者が約2か月、現場に入って設定する", { fontSize: 8.5, color: K.gray700 })], { x: rx + 0.2, y: top + 2.42, w: rw - 0.4, h: 0.48, size: 10 });
    box(s, { x: X0, y: 4.45, w: W, h: 0.62, fill: K.fill });
    txt(s, [run("たとえるなら、スマホの契約と同じ。", { bold: true, color: ACC, breakLine: true }), run("初期設定（導入支援）は最初だけ。月額料金（利用料）は毎月入ってくる → 去年のお客様が、今年も払ってくれる")], { x: X0 + 0.15, y: 4.45, w: W - 0.3, h: 0.62, size: 10.5, valign: "middle" });
  }

  // ---------- 6 お金の話2 なぜ払う ----------
  {
    const s = add(7);
    heading(s, "お金の話2　工事部長は、なぜ払うの？", "月25万円払うと、月240万円分の時間が浮く。約2か月で元が取れる");
    const top = 1.2;
    // equation chips
    const eq = [["100人", "施工管理者・電工"], ["×", ""], ["月6時間", "1人の書類作業が減る"], ["×", ""], ["4,000円", "1時間の人件費"], ["＝", ""], ["月240万円", "浮く時間の価値"]];
    let x = X0;
    eq.forEach(([a, b]) => {
      const op = b === "";
      const w = op ? 0.35 : a === "月240万円" ? 1.9 : 1.45;
      if (!op) box(s, { x, y: top, w, h: 0.9, fill: a === "月240万円" ? K.ink : K.fill });
      txt(s, a, { x, y: top + (op ? 0.15 : 0.08), w, h: op ? 0.6 : 0.5, size: op ? 22 : 18, bold: true, align: "center", valign: "middle", color: a === "月240万円" ? "E3A774" : K.ink });
      if (!op) txt(s, b, { x, y: top + 0.56, w, h: 0.28, size: 8.5, align: "center", color: a === "月240万円" ? K.paper : K.gray700 });
      x += w + 0.06;
    });
    // compare bars
    const by = top + 1.2, bx = X0 + 1.3, scale = 3.9 / 240;
    txt(s, "1か月あたり", { x: X0, y: by, w: 3, h: 0.26, size: 10, bold: true });
    [["払うお金", 25, K.gray500, "月25万円（初年度は導入支援込みで月42万円）"], ["浮くお金", 240, ACC, "月240万円"]].forEach(([n, v, c, lab], i) => {
      const y = by + 0.35 + i * 0.55;
      txt(s, n, { x: X0, y, w: 1.5, h: 0.42, size: 11, bold: true, valign: "middle" });
      box(s, { x: bx, y, w: v * scale, h: 0.42, fill: c });
      if (i === 1) txt(s, lab, { x: bx + 0.1, y, w: v * scale - 0.2, h: 0.42, size: 12, bold: true, valign: "middle", color: K.paper, align: "right" });
      else txt(s, lab, { x: bx + v * scale + 0.08, y, w: 4.0, h: 0.42, size: 10.5, valign: "middle" });
    });
    // fix label on the long bar: put inside
    ch(s, "manager-happy", 8.2, top + 1.25, 1.45);
    bubble(s, "2か月で戻るなら\n安い！", { x: 6.15, y: top + 1.05, w: 1.85, h: 0.75, tail: "br", size: 11, bold: true });
    box(s, { x: X0, y: 4.0, w: W, h: 1.08, fill: K.fill });
    txt(s, [
      run("元が取れるまで：", { bold: true, color: ACC }), run("初年度の支払い 500万円 ÷ 月240万円 ＝ 約2.1か月", { bold: true, breakLine: true }),
      run("月6時間って？", { bold: true, color: ACC }), run("　6時間÷20日＝1日あたり約18分。朝の日報と夕方の写真整理がなくなるイメージ", { breakLine: true }),
      run("1時間4,000円って？", { bold: true, color: ACC }), run("　年収600万円×1.3（社会保険など）÷年1,950時間"),
    ], { x: X0 + 0.15, y: 4.0, w: W - 0.3, h: 1.08, size: 10, valign: "middle", psa: 2 });
  }

  // ---------- 7 お金の話3 何が残る ----------
  {
    const s = add(8);
    heading(s, "お金の話3　うちの財布に、何が残る？", "利用料は85%が残る。導入支援は40%しか残らない。だから利用料を積み上げる");
    const base = 4.2, sc = 2.35 / 300;
    const bar = (x, v, fill, label, top0 = 0, ink = K.paper) => {
      box(s, { x, y: base - (top0 + v) * sc, w: 0.95, h: v * sc, fill, line: K.ink, lw: 0.75 });
      txt(s, label, { x: x - 0.2, y: base - (top0 + v) * sc - 0.32, w: 1.35, h: 0.3, size: 11, bold: true, align: "center", valign: "bottom", color: K.ink });
    };
    hline(s, X0, base, 5.4, K.ink, 1.5);
    // license
    txt(s, "利用料（毎年）", { x: X0, y: 1.15, w: 3.2, h: 0.28, size: 11, bold: true });
    bar(X0 + 0.1, 300, K.ink, "300万円");
    bar(X0 + 1.2, 45, K.gray500, "−45万円", 255);
    bar(X0 + 2.3, 255, ACC, "255万円");
    [["受け取る", 0.1], ["サーバー・AI代", 1.2], ["残る（粗利）", 2.3]].forEach(([t, dx]) => txt(s, t, { x: X0 + dx - 0.2, y: base + 0.05, w: 1.35, h: 0.25, size: 9, align: "center" }));
    // setup
    const sx = X0 + 3.5;
    txt(s, "導入支援（初年度だけ）", { x: sx - 0.1, y: 1.15, w: 2.5, h: 0.28, size: 11, bold: true });
    box(s, { x: sx, y: base - 200 * sc, w: 0.7, h: 200 * sc, fill: K.ink, line: K.ink, lw: 0.75 });
    txt(s, "200万円", { x: sx - 0.2, y: base - 200 * sc - 0.32, w: 1.1, h: 0.3, size: 11, bold: true, align: "center", valign: "bottom" });
    box(s, { x: sx + 0.85, y: base - 80 * sc, w: 0.7, h: 80 * sc, fill: K.gray500, line: K.ink, lw: 0.75 });
    txt(s, "80万円", { x: sx + 0.65, y: base - 80 * sc - 0.32, w: 1.1, h: 0.3, size: 11, bold: true, align: "center", valign: "bottom" });
    txt(s, "受け取る", { x: sx - 0.2, y: base + 0.05, w: 1.1, h: 0.25, size: 9, align: "center" });
    txt(s, "残る", { x: sx + 0.65, y: base + 0.05, w: 1.1, h: 0.25, size: 9, align: "center" });
    txt(s, "120万円は\n担当者の人件費", { x: sx + 0.8, y: base - 200 * sc, w: 1.2, h: 0.5, size: 8.5, color: K.gray700 });
    // character
    bubble(s, "人が動く仕事は残りが少ない。\nだから利用料を積み上げる\n会社にします！", { x: 6.15, y: 1.15, w: 2.5, h: 0.95, tail: "br", size: 10 });
    ch(s, "founder-happy", 8.35, 1.6, 1.25);
    box(s, { x: 6.15, y: 2.95, w: 3.45, h: 1.25, fill: K.fill });
    txt(s, [run("粗利（あらり）とは", { bold: true, color: ACC, breakLine: true }), run("売上から、そのサービスを作るのに直接かかったお金を引いた残り。ここから社員の給料や家賃を払う")], { x: 6.3, y: 3.0, w: 3.15, h: 1.15, size: 10, valign: "middle" });
    txt(s, "1拠点あたりの金額。サーバー・AI代は売上の15%、導入支援の人件費は売上の60%と置いた", { x: X0, y: 4.75, w: W, h: 0.26, size: S.note, color: K.gray500 });
  }

  // ---------- 8 お金の話4 2.8 ----------
  {
    const s = add(9);
    heading(s, "お金の話4　なぜ最初は赤字？ 合言葉は「2.8」", "社員1人で2.8拠点を見られれば黒字。4年目に超える");
    const top = 1.2;
    // equation
    box(s, { x: X0, y: top, w: 4.3, h: 1.55, fill: K.fill });
    txt(s, [run("社員1人にかかる費用", { breakLine: true }), run("年713万円", { bold: true, fontSize: 20 })], { x: X0 + 0.1, y: top + 0.1, w: 2.0, h: 0.8, size: 9.5, align: "center" });
    txt(s, "÷", { x: X0 + 2.0, y: top + 0.25, w: 0.3, h: 0.5, size: 22, bold: true, align: "center" });
    txt(s, [run("1拠点から残る粗利", { breakLine: true }), run("年255万円", { bold: true, fontSize: 20 })], { x: X0 + 2.25, y: top + 0.1, w: 2.0, h: 0.8, size: 9.5, align: "center" });
    txt(s, [run("＝ "), run("2.8拠点", { color: ACC, fontSize: 26 })], { x: X0, y: top + 0.88, w: 4.3, h: 0.6, size: 18, bold: true, align: "center", valign: "middle" });
    ch(s, "founder-normal", X0 + 0.1, top + 1.75, 1.3);
    [0, 1, 2].forEach((i) => ic(s, "office", X0 + 1.35 + i * 0.75, top + 2.1, 0.65));
    box(s, { x: X0 + 1.35 + 2 * 0.75 + 0.52, y: top + 2.0, w: 0.16, h: 0.8, fill: K.paper }); // trims the 3rd office to read as 2.8
    txt(s, "1人で、2.8拠点（支店・営業所）を見る", { x: X0 + 1.3, y: top + 2.85, w: 2.9, h: 0.28, size: 10, bold: true });
    // bars: sites per staff
    const gx = 5.05, gw = 4.55, base = 4.15, sc = 2.3 / 4.5;
    txt(s, "社員1人あたりの拠点数", { x: gx, y: top, w: gw, h: 0.26, size: 10.5, bold: true });
    hline(s, gx, base, gw, K.ink, 1.5);
    R.forEach((r, i) => {
      const v = r.sitesPerStaff, x = gx + 0.2 + i * 0.86, over = v > 2.8;
      box(s, { x, y: base - v * sc, w: 0.55, h: v * sc, fill: over ? ACC : K.ink });
      txt(s, v.toFixed(2), { x: x - 0.15, y: base - v * sc - 0.26, w: 0.85, h: 0.24, size: 10, bold: true, align: "center" });
      txt(s, r.year + "年目", { x: x - 0.15, y: base + 0.04, w: 0.85, h: 0.22, size: 9, align: "center" });
      txt(s, `${r.sites}拠点÷${r.staff}人`, { x: x - 0.2, y: base + 0.25, w: 0.95, h: 0.2, size: 7.5, align: "center", color: K.gray700 });
    });
    s.addShape("line", { x: gx, y: base - 2.8 * sc, w: gw, h: 0, line: { color: ACC, width: 1.75, dashType: "dash" } });
    txt(s, "黒字ライン 2.8", { x: gx + 0.05, y: base - 2.8 * sc - 0.27, w: 1.3, h: 0.24, size: 9.5, bold: true, color: ACC });
    bubble(s, "1〜3年目は人を先に\n雇うから赤字。\n4年目に線を越えて黒字！", { x: gx + 0.05, y: top + 0.32, w: 2.3, h: 0.8, tail: "bl", size: 9.5 });
    txt(s, "社員1人の費用＝4年目の販管費1億700万円÷15人（給料のほか家賃・広告・交通費などを含む）", { x: X0, y: 4.75, w: W, h: 0.26, size: S.note, color: K.gray500 });
  }

  // ---------- 9 お金の話5 雪だるま ----------
  {
    const s = add(10);
    heading(s, "お金の話5　去年のお客様が今年も払う。売上は雪だるま式", "5年目の売上2億9,000万円のうち57%は、前の年までのお客様の利用料");
    const years = R.map((r) => r.year + "年目");
    const ser = [
      ["前の年までのお客様の利用料", R.map((r) => r.existing / 100), K.ink],
      ["新しいお客様の利用料", R.map((r) => r.newLic / 100), K.gray700],
      ["導入支援", R.map((r) => r.setup / 100), K.gray500],
      ["受託開発", R.map((r) => r.contract / 100), K.rule],
    ];
    s.addChart(pres.charts.BAR, ser.map(([name, values]) => ({ name, labels: years, values })), {
      x: X0, y: 1.15, w: 5.6, h: 3.5, barDir: "col", barGrouping: "stacked", chartColors: ser.map((x) => x[2]), barGapWidthPct: 60,
      catAxisLabelColor: K.gray700, valAxisLabelColor: K.gray700, catAxisLabelFontSize: 9, valAxisLabelFontSize: 8, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
      valGridLine: { color: K.rule, size: 0.5 }, valAxisMaxVal: 300, valAxisMajorUnit: 50, valAxisLineShow: false,
      showLegend: true, legendPos: "t", legendFontSize: 8, legendFontFace: "+mn-lt", objectName: "snowball",
    });
    const rx = 6.25, rw = 3.35;
    txt(s, "売上（百万円）", { x: rx, y: 1.15, w: rw, h: 0.26, size: 10.5, bold: true });
    L.table(s, [["", "売上", "営業利益"], ...R.map((r) => [r.year + "年目", String(M(r.sales)), { text: L.tri(M(r.op)), color: r.op < 0 ? K.gray700 : ACC, bold: r.op > 0 }])],
      { x: rx, y: 1.45, w: rw, colW: [1.15, 1.1, 1.1], rowH: 0.3, size: 10, name: "sales-op" });
    ch(s, "founder-happy", rx + 2.05, 3.35, 1.3);
    bubble(s, "新しいお客様が\n増えるたびに\n大きくなる！", { x: rx, y: 3.4, w: 1.95, h: 0.95, tail: "br", size: 10 });
    txt(s, "新しいお客様は年の途中で始めるので、初年度の利用料は半年分で計算。1年目の3拠点は試験導入（利用料なし）", { x: X0, y: 4.75, w: W, h: 0.26, size: S.note, color: K.gray500 });
  }

  // ---------- 10 お金の話6 LTV/CAC ----------
  {
    const s = add(11);
    heading(s, "お金の話6　64万円のエサで、1,865万円の魚を釣る", "1拠点を連れてくる費用（CAC）と、1拠点が残してくれる粗利（LTV）");
    const top = 1.2;
    box(s, { x: X0, y: top, w: 3.6, h: 2.85, line: K.ink, lw: 1.25 });
    ic(s, "bait", X0 + 0.15, top + 0.2, 1.1);
    txt(s, [run("連れてくる費用", { breakLine: true }), run("CAC（キャック）", { fontSize: 9, color: K.gray700 })], { x: X0 + 1.3, y: top + 0.25, w: 2.2, h: 0.55, size: 12, bold: true });
    txt(s, "約64万円", { x: X0 + 1.3, y: top + 0.8, w: 2.2, h: 0.5, size: 24, bold: true });
    txt(s, [run("（営業2人の人件費1,600万円", { breakLine: true }), run("＋広告・展示会500万円）", { breakLine: true }), run("÷ 新しいお客様33拠点（5年目）")], { x: X0 + 0.15, y: top + 1.55, w: 3.3, h: 0.9, size: 10 });
    txt(s, "約29倍", { x: 4.05, y: top + 0.85, w: 1.5, h: 0.55, size: 24, bold: true, color: ACC, align: "center" });
    arrow(s, 4.1, top + 1.5, 1.4, 0, { lw: 3, color: ACC });
    box(s, { x: 5.6, y: top, w: 4.0, h: 2.85, fill: K.ink });
    ic(s, "fish", 5.75, top + 0.2, 1.1);
    txt(s, [run("残してくれる粗利", { breakLine: true }), run("LTV（エルティーブイ）", { fontSize: 9 })], { x: 6.95, y: top + 0.25, w: 2.55, h: 0.55, size: 12, bold: true, color: K.paper });
    txt(s, "約1,865万円", { x: 6.95, y: top + 0.8, w: 2.6, h: 0.5, size: 24, bold: true, color: "E3A774" });
    txt(s, [run("年255万円 × 7年", { breakLine: true }), run("＋ 導入支援の粗利80万円", { breakLine: true }), run("7年は控えめ。計画の解約率なら平均18年続く", { fontSize: 9 })], { x: 5.75, y: top + 1.55, w: 3.7, h: 0.95, size: 10.5, color: K.paper });
    box(s, { x: X0, y: 4.25, w: W, h: 0.8, fill: K.fill });
    ch(s, "client-happy", X0 + 0.1, 4.28, 0.75);
    txt(s, [run("64万円は約3か月で回収（64万円 ÷ 月21万円の粗利）。", { bold: true, breakLine: true }), run("目安は「LTVがCACの3倍以上」。当社は約29倍なので、お客様を増やすほど会社が強くなる")], { x: X0 + 0.85, y: 4.25, w: W - 1.0, h: 0.8, size: 10.5, valign: "middle" });
  }

  // ---------- 11 500万円 ----------
  {
    const s = add(12);
    heading(s, "500万円で、6か月以内に最初の1拠点を動かす", "創業者の給料はゼロ。すべて試作品と実証に使う");
    const top = 1.2;
    const steps = [["今日", "500万円の出資", "試作品・端末・実証"], ["3か月", "試作品ができる", "記録→写真抽出→報告書"], ["6か月", "1拠点で実証", "報告書の時間がどれだけ減ったか実測"], ["その後", "公庫1,000万円＋VC1.15億円", "実証の結果を持って調達"]];
    steps.forEach(([t, a, b], i) => {
      const x = X0 + i * 2.33;
      box(s, { x, y: top, w: 2.15, h: 1.5, fill: i === 0 ? K.ink : K.fill, line: i === 0 ? undefined : K.rule });
      const c = i === 0 ? K.paper : K.ink;
      txt(s, t, { x: x + 0.12, y: top + 0.1, w: 1.9, h: 0.3, size: 11, bold: true, color: i === 0 ? "E3A774" : ACC });
      txt(s, a, { x: x + 0.12, y: top + 0.45, w: 1.95, h: 0.5, size: 12, bold: true, color: c });
      txt(s, b, { x: x + 0.12, y: top + 0.98, w: 1.95, h: 0.45, size: 9, color: c });
      if (i < 3) arrow(s, x + 2.17, top + 0.75, 0.15, 0, { lw: 2 });
    });
    const uses = [["試作品の開発", 250], ["実証（1拠点・3か月）", 100], ["端末・機材", 80], ["設立・契約", 40], ["予備", 30]];
    txt(s, "使い道（万円）", { x: X0, y: 2.95, w: 4, h: 0.26, size: 10.5, bold: true });
    const k = 2.4 / 250;
    uses.forEach(([n, v], i) => {
      const y = 3.27 + i * 0.33;
      txt(s, n, { x: X0, y, w: 1.75, h: 0.28, size: 9.5, valign: "middle" });
      box(s, { x: X0 + 1.8, y: y + 0.04, w: v * k, h: 0.2, fill: i === 0 ? ACC : K.ink });
      txt(s, String(v), { x: X0 + 1.85 + v * k, y, w: 0.5, h: 0.28, size: 10, bold: true, valign: "middle" });
    });
    box(s, { x: 5.0, y: 2.95, w: 4.6, h: 1.95, line: ACC, lw: 2 });
    ch(s, "founder-normal", 5.15, 3.2, 1.4);
    txt(s, [run("虎へのお返し（提案）", { bold: true, color: ACC, fontSize: 11, breakLine: true }), run("500万円で株式の10%", { bold: true, fontSize: 14, breakLine: true }), run("（出資前の会社の値段 4,500万円）", { fontSize: 9, breakLine: true }), run("出口：上場、または事業会社への売却", { fontSize: 9.5, breakLine: true }), run("お願い：実証先の電気工事会社のご紹介も", { fontSize: 9.5 })], { x: 6.4, y: 3.05, w: 3.1, h: 1.75, size: 10, valign: "middle", psa: 2 });
  }

  // ---------- 12 全体図 ----------
  {
    const s = add(13);
    heading(s, "1枚で言うと ─ 上は記録とサービス、下はお金", "記録が貯まるほど真似されにくく、お客様が増えるほど黒字が厚くなる");
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
    txt(s, "三現ワークス", { x: cx, y: top + 0.45, w: cw, h: 0.3, size: 12, bold: true, color: K.paper, align: "center" });
    ic(s, "server", cx + 0.95, top + 0.85, 0.7);
    txt(s, [run("AIが整理", { bold: true, breakLine: true }), run("記録が貯まるほど", { breakLine: true }), run("真似されにくくなる", { color: "E3A774", bold: true })], { x: cx + 0.1, y: top + 1.6, w: cw - 0.2, h: 0.85, size: 10, color: K.paper, align: "center" });
    txt(s, [run("1拠点の粗利 255万円", { bold: true, breakLine: true }), run("（300万円 − サーバー代45万円）")], { x: cx + 0.1, y: top + 2.55, w: cw - 0.2, h: 0.6, size: 9.5, color: K.paper, align: "center" });

    // top arrows: service
    arrow(s, 2.6, top + 0.75, 1.05, 0, { lw: 2.5 });
    txt(s, "① 動画と会話", { x: 2.5, y: top + 0.42, w: 1.25, h: 0.28, size: 9, bold: true, align: "center" });
    arrow(s, 2.6, top + 1.6, 1.05, 0, { lw: 2.5, flipH: true });
    txt(s, "② 照合・報告書\n・教材", { x: 2.5, y: top + 1.05, w: 1.25, h: 0.5, size: 9, bold: true, align: "center" });
    // bottom arrow: money
    arrow(s, 2.6, top + 2.95, 1.05, 0, { lw: 4, color: ACC });
    ic(s, "coin", 2.9, top + 2.3, 0.5);
    txt(s, "③ 年300万円", { x: 2.5, y: top + 3.05, w: 1.25, h: 0.28, size: 10, bold: true, color: ACC, align: "center" });

    // right: costs
    const kx = 6.65, kw = 2.95;
    arrow(s, cx + cw + 0.04, top + 1.05, 0.38, 0, { lw: 2.5 });
    arrow(s, cx + cw + 0.04, top + 2.75, 0.38, 0, { lw: 2.5 });
    box(s, { x: kx, y: top + 0.35, w: kw, h: 1.4, fill: K.fill });
    txt(s, [run("④ 出ていくお金", { bold: true, color: ACC, breakLine: true }), run("サーバー・AI代：売上の15%", { breakLine: true }), run("社員1人：年713万円（給料・家賃など）")], { x: kx + 0.12, y: top + 0.4, w: kw - 0.24, h: 1.3, size: 10, valign: "middle", psa: 3 });
    box(s, { x: kx, y: top + 2.0, w: kw, h: 1.55, line: ACC, lw: 2 });
    txt(s, [run("⑤ 黒字の条件", { bold: true, color: ACC, breakLine: true }), run("社員1人 × 2.8拠点 以上", { bold: true, fontSize: 15, breakLine: true }), run("4年目に3.67となり黒字", { breakLine: true }), run("5年目 営業利益 6,900万円")], { x: kx + 0.12, y: top + 2.05, w: kw - 0.24, h: 1.45, size: 10, valign: "middle", psa: 2 });
    txt(s, "将来：貯まった作業の記録で、ロボットに作業を教える（フィジカルAI）", { x: X0, y: 4.85, w: W, h: 0.26, size: 9.5, bold: true, color: K.gray700 });
  }

  // ---------- 13 ツッコミ対策 ----------
  {
    const s = add(14);
    heading(s, "虎のツッコミ3つ ─ この一言で返す", "答えは必ず数字から言う。理由は後");
    const qa = [
      ["本当に年300万円も払うの？", "月240万円分の時間が浮くので、約2か月で元が取れます", "100人×月6時間×4,000円"],
      ["なんで4年目まで赤字なの？", "人を先に雇うからです。社員1人2.8拠点を超えたら黒字です", "713万円÷255万円＝2.8"],
      ["真似されたら終わりでは？", "現場の記録は使うほど貯まり、後から来た会社には移せません", "貯まる・意味づけ・業界の型"],
    ];
    qa.forEach(([q, a, why], i) => {
      const y = 1.15 + i * 1.25;
      ch(s, "client-think", X0, y, 1.05);
      bubble(s, q, { x: X0 + 0.95, y: y + 0.1, w: 2.6, h: 0.6, tail: "bl", size: 10.5, bold: true });
      bubble(s, a, { x: 4.05, y: y + 0.1, w: 4.45, h: 0.75, tail: "br", size: 10.5, fill: K.fill });
      ch(s, "founder-happy", 8.6, y, 1.05);
      txt(s, "根拠：" + why, { x: 4.1, y: y + 0.88, w: 4.1, h: 0.24, size: 8.5, color: ACC, bold: true });
    });
    txt(s, "左：虎（投資家）　右：社長（わたし）。細かい計算は「理論武装版」に", { x: X0, y: 4.9, w: W, h: 0.24, size: S.note, color: K.gray500 });
  }

  // ---------- 14 まとめ ----------
  {
    const s = add(15);
    heading(s, "まとめ ─ この3つの数字だけ覚えれば、話せる", "300万・255万・2.8。順番に言えば、お金の仕組みが全部つながる");
    const top = 1.15;
    const big = [["300万円", "お客様が1拠点で毎年払う", "2か月で元が取れるから払う"], ["255万円", "そのうち当社に残る粗利", "サーバー代15%を引いた残り"], ["2.8拠点", "社員1人で見れば黒字", "4年目に超え、5年目は4.25"]];
    big.forEach(([n, a, b], i) => {
      const x = X0 + i * 3.1;
      box(s, { x, y: top, w: 2.95, h: 1.55, fill: i === 2 ? K.ink : K.fill });
      const c = i === 2 ? K.paper : K.ink;
      txt(s, n, { x: x + 0.15, y: top + 0.1, w: 2.7, h: 0.65, size: 30, bold: true, color: i === 2 ? "E3A774" : ACC, valign: "middle" });
      txt(s, a, { x: x + 0.15, y: top + 0.78, w: 2.7, h: 0.3, size: 11.5, bold: true, color: c });
      txt(s, b, { x: x + 0.15, y: top + 1.1, w: 2.7, h: 0.3, size: 9.5, color: c });
      if (i < 2) arrow(s, x + 2.97, top + 0.78, 0.11, 0, { lw: 2 });
    });
    box(s, { x: X0, y: 2.9, w: W, h: 1.95, line: K.ink, lw: 1.25, name: "script30" });
    txt(s, "30秒で言うと（そのまま読める台本）", { x: X0 + 0.15, y: 2.97, w: 5, h: 0.28, size: 10.5, bold: true, color: ACC });
    txt(s, "電気工事会社が1拠点あたり年300万円を払います。月240万円分の書類の時間が浮くので、2か月で元が取れます。当社にはサーバー代を引いた255万円が残り、社員1人で2.8拠点を見れば黒字です。人を先に雇うので3年目までは赤字ですが、去年のお客様が今年も払う積み上げ型なので、4年目に黒字、5年目に営業利益6,900万円になります。", { x: X0 + 0.15, y: 3.28, w: W - 1.6, h: 1.5, size: 11, lsm: 1.3 });
    ch(s, "founder-happy", 8.35, 3.3, 1.35);
  }

  fs.mkdirSync(OUT_DIR, { recursive: true });
  await pres.writeFile({ fileName: OUT });
  console.log("wrote", OUT);
}

main().catch((e) => { console.error(e); process.exit(1); });
