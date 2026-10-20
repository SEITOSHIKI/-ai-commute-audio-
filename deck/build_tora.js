// Build the business plan deck for KANTSUGI (カンツギ): 33 slides in the order of the assignment's questions ①〜⑤
// (①事業・理由・想い ②スタッフ ③経営理念 ④年単位の計画と収支 ⑤段階ごとの課題).
// Usage: node build_deck.js [outDir]   (needs pptxgenjs and sharp)
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const sharp = require("sharp");
const T = require("../design-system/georec/pptx-theme.js");

const { color: K, size: S } = T;
const OUT_DIR = process.argv[2] || __dirname;
const OUT = path.join(OUT_DIR, "カンツギ_事業計画書_本編.pptx");
const { customerValue, nationValue } = require("./tora_value.js");
const { run: model, unit, shortfall, SALES } = require("./tora_model.js");
const PLAN = model(), PR = PLAN.rows, U = unit();
const Y5 = PR[4], fb = PLAN.firstBlack;
// 計算詳細版（質問用）のページ。本編の数字の式と前提はそこにある
const DP = { value: "5〜8", price: "9〜13", device: "10・26・29", sales: "14〜16", team: "17", pl: "18", bep: "19", ltv: "20", gates: "21〜22", cash: "23", market: "24", target: "25", trial: "26", data: "27", ml: "28", lease: "29" };
const mil = (v) => Math.round(v / 100); // 万円 → 百万円
const CV = customerValue(), NV = nationValue();
const V = require("./tora_v2.js");
const L = require("./tora_lib.js").drawing(); // 16枚版と同じ見た目（K を紺・安全色に切り替える）
const TH = require("./tora16_theme.js");
const man = (v) => Math.round(v).toLocaleString("en-US");
const oku = (v) => (v >= 10000 ? Math.floor(v / 10000) + "億" + (v % 10000 ? man(v % 10000) + "万円" : "円") : man(v) + "万円");
const be = Math.round(CV.breakEvenMin); // お客様が元を取る1台1日の分数
const gapSites = Math.ceil((Y5.sga - Y5.gp) / U.gpYear); // 5年目に黒字まで足りない拠点数
const D15 = V.run({ plan: { devices: 15, mix: { glass: 12, display: 3 } } });
const R10 = V.run({ years: 10 }), R10_15 = V.run({ years: 10, plan: { devices: 15, mix: { glass: 12, display: 3 } } });
const TEN = [["計画どおり（10台）", {}], ["1拠点15台", { plan: { devices: 15, mix: { glass: 12, display: 3 } } }], ["解約 年20%", { churnRate: 0.2 }], ["値下げ −20%", { priceK: 0.8 }],
  ["新規 −30%（採用も連動）", { newSites: V.GROWTH.newSites.concat(V.EXT.newSites).map((n) => Math.round(n * 0.7)), contract: V.GROWTH.contract.concat(V.EXT.contract).map((c) => c * 0.7), hireK: 0.7 }],
  ["AI・保存費が2倍", { perDevice: V.cloudPerDevice * 2 }]].map(([n, o]) => { const r = V.run({ years: 10, ...o }); return { n, fb: r.firstBlack, cb: r.cumBack, min: r.minCash }; });
const IMPACT = (() => { const siteYears = R10.rows.reduce((a, r) => a + r.sites, 0); return { siteYears, hours: siteYears * CV.freedHours, value: siteYears * CV.docs }; })();
const LOGOS = path.join(__dirname, "../design-system/georec/logos");
const F = "BIZ UDPGothic";
const X0 = 0.4, W = 9.2;

async function logo(name) {
  const buf = await sharp(path.join(LOGOS, name + ".svg"), { density: 600 }).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

// ---------- primitives (fresh option objects every call) ----------
function txt(slide, text, o) {
  slide.addText(text, {
    fontFace: F, fontSize: o.size || S.body, color: o.color || "1F2933", bold: !!o.bold,
    x: o.x, y: o.y, w: o.w, h: o.h, margin: o.margin ?? 0, valign: o.valign || "top",
    align: o.align || "left", lineSpacingMultiple: o.lsm || 1.15, paraSpaceAfter: o.psa || 0,
    fill: o.fill ? { color: o.fill } : undefined, isTextBox: true, objectName: o.name,
  });
}
function box(slide, o) {
  slide.addShape("rect", {
    x: o.x, y: o.y, w: o.w, h: o.h,
    fill: { color: o.fill || K.paper },
    line: o.line ? { color: o.line, width: o.lw || T.stroke.rule, dashType: o.dash } : { type: "none" },
    objectName: o.name,
  });
}
function hline(slide, x, y, w, c, lw) {
  slide.addShape("line", { x, y, w, h: 0, line: { color: c, width: lw } });
}
function arrow(slide, x, y, w, h, o = {}) {
  slide.addShape("line", { x, y, w, h, line: { color: o.color || K.ink, width: o.lw || 2, endArrowType: "triangle" } });
}
const run = (text, o = {}) => ({ text, options: { fontFace: F, ...o } });

const ACC = "2F6DB5"; // 青（16枚版と同じ）。赤字は RED、黒字・進むは GREEN
function detailRef(slide, pages) { // 右上に「詳細版 p.○」を出す
  txt(slide, "計算の詳細：詳細版 p." + pages, { x: 6.6, y: 0.13, w: 3.0, h: 0.16, size: 7.5, color: K.gray500, align: "right", valign: "middle" });
}
// 見出しと表は16枚版（tora16_theme.js）と同じ。見出しは「項目：結論」
function heading(slide, title, lead) {
  TH.heading(slide, typeof title === "string" ? title.replace(" ─ ", "：") : title, lead);
}

// 表：紺の見出し行。emCol の列は薄い青、mutedCol の列は灰色の文字
function table(slide, rows, o) {
  const prep = rows.map((r, i) => r.map((c, j) => {
    if (i === 0) return c;
    const cell = typeof c === "object" ? { ...c } : { text: String(c) };
    if (o.emCol === j && !cell.fill) cell.fill = TH.PALE;
    if (o.mutedCol === j && !cell.color) cell.color = K.gray700;
    return cell;
  }));
  TH.table(slide, prep, o);
}

const TAGLINE = "現場の勘を次の担い手へ";

// ---------- notes: time slot, the slide's one message, then the script ----------
const NOTE_LIST = [
  null,
  ["人が減っても現場が迷わず回る仕組みをつくる",
    "株式会社カンツギです。カンツギは「勘を継ぐ」。割れた器を金で継ぐ金継ぎのように、途切れそうな現場の勘をデータで継ぎます。熟練者の勘が退職とともに消えていく現場に、普段のメガネと同じ軽さのAIグラスで作業を記録し、次の担い手とAIに渡す仕組みをつくります。"],
  ["課題の設問①〜⑤の順に話す",
    "本日は、課題の5つの設問の順にお話しします。①どんな事業か、選んだ理由と想い。②どんなスタッフと働くか。③経営理念。④年単位の経営計画と収支。⑤各段階で想定される課題です。各ページの左上の番号が設問の番号です。前提は、来年、豊富な資金をもとに事業を立ち上げること。資金は自己資金でまかないます。計算の過程は別冊の計算詳細版にあり、右上にページを書いています。"],
  ["AIグラスで工事の作業を記録し、報告と教育を自動にするサービス",
    "事業の概要です。何のビジネスか。職長や施工管理者がAIグラスを掛けるだけで作業を記録し、工事写真の整理と報告書づくり、若手の教育を自動にするサービスです。既存のサービスとの違いは、人が撮って入力するのではなく自動で記録し、図面と照合し、教材にまですること。お客様は、関東の中堅の電気工事会社です。収益は、会社の基盤と端末の台数に応じた利用料、標準の10台で年300万円と、初年度の導入支援です。"],
  ["現場の勘が消えるのを見てきた。市場・技術・自分の3つの理由",
    "なぜこの事業を選んだか。理由は3つです。市場の理由。建設業は55歳以上が36.7%、2024年から残業に上限があり、書類と育成に回す人がいません。お金を払ってでも解きたい問題があります。技術の理由。見て理解するAIが実用になり、カメラ付きのAIグラスが普段のメガネと同じ約50グラムになりました。一日中掛けていられる時代が来ています。そして自分の理由。私は電線メーカーで地中送電線の施工管理を5年やりました。ベテランの判断が手順書に書かれないまま、退職とともに消えていくのを見てきました。現場とAIの両方がわかる人は多くありません。だから私がやります。想いは社名の通りです。途切れそうな現場の勘を金継ぎのように継いで、人が減っても日本の現場の品質を落とさない。"],
  ["AIは事務作業の支援から、作業者の「目と手」の支援へ", "背景です。これまでAIは、管理者の報告書や議事録といった事務作業の効率化に使われてきました。考える・書くAIは、もう十分に実用の水準です。次は、これまで代わりが利かなかった肉体労働、とくに作業者の暗黙知を支える番です。安全・品質・施工手順の作業支援。人の目は口ほどに物を言うと言いますが、次に来るのは「見るAI」だと考えています。その入口が、普段のメガネと同じ軽さのAIグラスです。スマホ世代の次の、ARグラス世代が憧れる施工のやり方をつくります。"], ["残業の上限と高齢化で、書類と育成に回す人がいない", "対象のお客様は三人です。一人目は、現場で孤軍奮闘するDX推進担当者。二人目は、施工管理者と電工。工事写真の整理と報告書に毎日追われ、聞ける熟練者もいない。三人目は、お金を出す工事部長や社長。2024年4月から建設業にも残業の上限がかかり、人を増やせない。建設業で働く人の36.7%が55歳以上、29歳以下は11.7%しかいません。"], ["手戻りだけで建設全体に年1.5兆円。勘の喪失は国の損失", "思いだけの話ではありません。お金で見ると、建設投資は年約75兆円。手戻り、つまりやり直しの費用は、海外の研究で工事費の0.4%から6%と言われています。控えめに2%とすると、建設全体で年約1.5兆円、電気工事だけで年約2,400億円がやり直しに消えています。そして建設業で働く人の36.7%、約175万人が55歳以上です。熟練者の判断が消えると、この手戻りが増え、残業の上限の中で受けられる工事も減ります。現場の勘を残すことは、この損失を防ぐことです。"], ["記録は自動、渡す先は三つ。データが貯まるほど真似されにくくなる", "解決策です。作業者の一人称の動画と会話を撮り続け、AIが整理し、若手には現物に重ねて、管理者には報告書として、次の担い手には教材として渡します。真似されにくい理由は三つ。図面に現れない現場処置の記録は、使うほど貯まり、後から来た会社には移せません。映像のどこが重要かを見分ける意味づけは、施工の経験者にしかできません。そして、電気工事の工種ごとの手順と報告書の型を、業界で最初に揃えます。"],
  ["1拠点10台で年約" + man(Math.round(CV.docs / 10) * 10) + "万円。支払いの約" + CV.ratio.toFixed(1) + "倍。1台1日" + be + "分減れば元が取れる",
    `お客様の効果です。効果は、AIグラスを掛けた班の分だけで数えます。1拠点に10台、1班に1台を職長が掛けます。1台で1日、職長の日報と撮影が15分、施工管理者の写真整理と写真台帳が20分減ると、統計から出した時給で、年約${man(Math.round(CV.docs / 10) * 10)}万円です。お客様が払うのは、当社への年300万円とグラスのリース約${man(V.leaseSite())}万円で、約${man(CV.cost)}万円。得はその約${CV.ratio.toFixed(1)}倍です。逆に言うと、1台1日${be}分減れば元が取れます。ここを実証の合格ラインにします。手戻りの削減は、実証で測れたら上乗せし、そのときは約${CV.ratioUp.toFixed(1)}倍です。新人の指導は金額にせず、独り立ちまでの月数で測ります。計算の過程は詳細版の5〜8ページにあります。`],
  ["壁の図面を、手元に実物大で出す。ミスしやすい所では止める", "ARグラスで何が変わるか。今の現場では、2メートル近いケーブルのジョイントの組立図を、わら半紙で壁に貼って施工しています。壁を見に行き、手元に戻り、寸法を思い出して測る。この往復が、思い込みと記憶違いのミスを生みます。ARなら、実物大・立体・工程ごとの完成形を手元に出せます。過去のミス事例を学習させておけば、間違えやすい所で、赤を切るか青を切るかのように、音声とレーザーポインターで止められます。グラスは火花や異物から目を守る役目も果たします。ただし寸法の最終確認はスケールで行います。測る道具ではなく、理解して間違いに気づく道具です。"], ["ベテランが解釈を教え、AIが育つ。人が抜けても資産が残る", "AIが育つ仕組みです。まず一人称で撮る。ベテランの管理者が、この場面が大事だと口頭で教える。AIが必要な場面を切り取って報告する。意味のあるデータに絞って集める。それを別の作業へ広げる。このループが回るほど賢くなります。その結果、報連相が口頭から自動に変わり、危険箇所の注意や、30分作業したら休憩の声かけ、VRの施工シミュレーターにも使えます。作業者が抜けても、管理者が抜けても、現場を解釈できるAIという資産が残ります。"], ["デモ：図面との差異を見つけ、報告書が自動でできる", "実際の画面です。ケーブル接続の現場で、若手が処理した半導電層が図面より8ミリ短いことを検出しました。遠隔の熟練者が、テープの位置から測り直すよう指示します。その判断も記録されます。管理者の画面では、手順ごとに写真が自動で割り当てられ、報告書ができていきます。作業者が報告書に使った時間はゼロです。"],
  ["普段のメガネの軽さのAIグラスを、1班に1台、職長が掛ける",
    "端末と使い方です。当社はグラスそのものは作りません。標準は、普段のメガネと同じ約50グラムのカメラ付きAIグラスです。誰が掛けるか。電工全員ではなく、1班に1台、職長が掛けます。1拠点で5現場に2班ずつ、10台です。どこで撮るか。作業エリアだけで、休憩や移動中は撮りません。何を撮るか。録画は端末に残し、作業の区切り、声の合図、検査の場面だけを短く切り出してAIに送ります。手元に図面を出す班だけ、表示付きのARグラスを使います。粉じんや高所の厳しい現場では、RealWearのような産業用のヘッドセットも選べるようにします。端末が進化したら乗り換えられる設計にし、当社はアプリとAIとデータ基盤を作ります。"],
  ["施工管理アプリは「人が入力する器」、産業用ヘッドセットは「人が見て判断」。当社は「自動で記録し、判断を渡す」",
    "競合です。工事写真や施工管理のアプリは、ANDPAD、SPIDERPLUS、蔵衛門など、すでに広く使われています。ただ、どれも人が撮って入力する器です。もう一つの競合が、RealWearのような産業用のヘッドセットと、遠隔臨場のウェアラブルカメラです。両手が空いたまま映像を記録し、遠くの熟練者とつなぐ点で、私たちと方向は近い。ただ、映像を見て判断するのは人で、報告書や教材にはなりません。RealWearも音声で操作するAIを載せ始めており、日立などの大手もウェアラブルカメラと生成AIの書類作成支援を開発しています。最大の脅威は、こうした会社が自動報告を加えることです。だから競わずにつなぎます。端末はRealWearも選べるようにし、報告書は各社の形式で出力します。当社は、電気工事の判断の記録と、施工経験者の監修で選ばれます。"],
  ["日立AI Safetyと方向は同じ。だから施工・中堅・書類と技能継承に領域をずらす", "先にお伝えすると、ほぼ同じ方向のサービスがあります。日立製作所、日立ビルシステム、日立パワーソリューションズが2025年10月に発表したAI Safetyです。ウェアラブルカメラの映像をAIが解析し、安全アラート、チャットボット、作業の案内、報告書の作成までを目指しています。11月からは日立ビルシステムの全国の現場で、数千人の技術者が使い始めました。技術と規模では、私たちは勝てません。だから領域をずらします。日立はビル設備の保守と安全から入っています。私たちは中堅の電気工事会社の、施工の現場から入ります。第一の価値は安全ではなく、写真台帳や検査記録といった書類の削減と、技能の継承です。報告書は電力会社や官公庁など発注者ごとの書式で出し、電気工事の工種の型を、施工経験者の監修で先に揃えます。もし日立が施工や中堅向けに来たら、競わずに提携も選びます。勝負の窓は2〜3年と見て、型とデータを先に貯めます。"],
  ["5年目" + Y5.staff + "人。現場監修とコンサル部隊を社内に持つ。私は型と判断の基準をつくる",
    `どんなスタッフと働くかです。5年目に${Y5.staff}人の体制にします。核は3つの役割です。映像AIとデータ基盤をつくるCTOとエンジニア。電気工事の施工管理を10年以上経験し、映像のどこが大事かを意味づける現場監修。そして、お客様の現場に入って導入し、各社の報告書の形や既存のシステムとの連携を設定する導入コンサルタントです。この設定は外部に委託せず、社内のコンサル部隊が行います。そうすることで、各社の帳票の型が当社の知見として貯まり、製品に戻せます。私の役割は、最初の営業と現場監修の統括、そして理念と判断の基準をつくることです。私は電線メーカーで地中送電線の施工管理を5年やり、独学でAIの仕組みを作ってきました。足りないところは、社外の技術顧問と業界の顧問、グラスのメーカーや大学と組んで補います。`],
  ["現場の勘を次の担い手へ。監視ではなく、技術を残す仕組み",
    "経営理念です。ホームページで最初に伝える言葉は「現場の勘を次の担い手へ」です。使命は、熟練者の判断をデータで継ぎ、人が減っても日本の現場の品質を守ること。大切にすることは3つ。現場を主役にすること。作業の手を止めず、入力を増やさない。勘に敬意を払うこと。記録は本人の同意のもとで行い、誰の技術かを残す。そして数字で約束すること。効果を毎月測って見せ、効かなければ正直に言う。特に現場で働く方には、これは監視ではなく、あなたの技術を残して評価につなげる仕組みだと伝えます。"],
  ["ターゲット市場は約141億円。周辺市場は年7〜9%で伸びている", "市場規模です。全体市場として、国内の建設テック市場は建築分野だけで2030年度に約3,040億円、年7.4%の成長です。ターゲット市場は、業種と規模で分けて数えました。電気工事と電気通信工事の中堅以上、約3,100社。顧客数かける単価かける年1回で、約141億円です。5年目に81拠点、約2.4億円を取りに行きます。"], ["最初の顧客は関東の中堅電気工事会社。4つの物差しで選んだ", "なぜ最初が中堅の電気工事会社なのか。市場の大きさ、困りごとの強さ、会いに行きやすさ、波及効果の4つで比べました。中堅の電気工事会社は、市場が最も大きく、残業規制で困りごとが強く、私の施工管理の経歴がそのまま信用になります。お金を出すのは社長や工事部長、使うのは施工管理者、推進するのはDX担当。この三人それぞれに効く説明を用意します。"], ["誰に・どう実現し・どう稼ぐか。ビジネスモデルを3つに分ける", "ビジネスモデルを3つに分けます。戦略は、中堅電気工事会社に、書類の時間を減らし熟練者の判断を残す価値を届けること。オペレーションは、デモから2か月の導入、スマートグラスでの記録、AIの整理と当社の現場監修、報告書と教材の納品、月次の効果測定までの流れです。導入コンサルタント1人が年6拠点を立ち上げます。マネタイズは、会社の基盤が月5万円、端末1台が月2万円の利用料を年額前払いで、初年度だけ導入支援をいただきます。"],
  ["会社の基盤＋端末の台数で払う。お客様は初年度約" + Math.round((CV.cost + V.PLAN.setup) / (CV.docs / 12)) + "か月、2年目からは毎年プラス",
    `お金の流れです。お客様は、会社の基盤に月5万円、端末1台に月2万円を払います。標準の10台で年300万円。AIの出力を各社の報告書の形に整えるところまで、この利用料に含みます。初年度だけ導入支援200万円、グラスはお客様がリースで借ります。初年度は合わせて約${man(CV.cost + V.PLAN.setup)}万円、得が月約${man(CV.docs / 12)}万円なので約${Math.round((CV.cost + V.PLAN.setup) / (CV.docs / 12))}か月で元が取れ、2年目からは毎年、払う額より約${man(CV.docs - CV.cost)}万円多く得をします。当社に残るのは、AIと保存の費用を引いた1拠点年${man(U.gpYear)}万円。7年分と導入支援の粗利でLTVは約${man(U.ltv(7))}万円、獲得費用CACは約${man(U.cac)}万円で、約${Math.round(U.ltv(7) / U.cac)}倍です。`],
  ["最初の10社は、業界団体・展示会・提携で連れてくる", "売り方です。関東の中堅電気工事会社、約735社に、業界団体の勉強会、電設の展示会、スマートグラスのメーカーとの提携で会いに行きます。デモを見ていただき、3か月の有償試験導入から本契約へ。中でも、電力会社や官公庁の工事が多く、手戻りが高くつく工種があり、デジタル化の素地がある会社、約180社を一次ターゲットにします。会える率と成約率をそれぞれ見積もり、1社1.3拠点と大手1社の5拠点を足して約85拠点。解約を引いた計画は81拠点です。"],
  ["売上は拠点×月25万円で積み上がる。解約は年10%で引く",
    `売上と費用です。売上は、続いている拠点かける月25万円の利用料が積み上がる形で、解約は毎年10%で引いています。5年目の売上は約${man(Y5.sales / 100)}百万円、月にすると約${man(Y5.sales / 12)}万円。このほか、大手のお客様の基幹システムとの連携などの個社設定を、当社の導入コンサルタントが有償で行います。費用の中心は人件費で、役割ごとの年収に会社負担を足し、採用費と顧問料、AIの学習の計算費も入れています。所在地は川崎市を考えています。羽田・新幹線に近く、関東一円の電気工事会社に日帰りで出られます。`],
  ["10台のままでは5年目でも届かない。足りないのは約" + gapSites + "拠点分",
    `損益分岐点です。正直に言うと、1拠点10台のままでは5年目でも届きません。販管費を粗利率で割った損益分岐点売上は、5年目で約${man(Y5.bep / 100)}百万円、売上は約${man(Y5.sales / 100)}百万円で、その${Math.round(Y5.sales / Y5.bep * 100)}%です。足りないのは、1拠点の粗利${man(U.gpYear)}万円で数えて約${gapSites}拠点分。人で見ると、社員1人の販管費約${man(Y5.sga / Y5.staff)}万円を1拠点の粗利で割って約${(Y5.sga / Y5.staff / U.gpYear).toFixed(1)}拠点。新しい拠点は初年度に半年分しか払わないので、期末で1人約4拠点が目安ですが、5年目は${Y5.sitesPerStaff.toFixed(2)}です。人を先に雇う計画なので、ここを越えるのは、1拠点の台数を増やしたときか、採用を抑えた6年目以降です。`],
  ["自己資金を段階に分けて入れる。線を越えたときだけ次を入れる",
    `資金計画です。課題の前提どおり、自己資金でまかないます。準備期に500万円、実証に合格したら1年目に2億円、3年目に撤退ラインを越えていれば5,000万円を追加し、合計約2.55億円です。5年間の赤字の累計が最も深くて約${(-PLAN.cumPeak / 1e4).toFixed(1)}億円なので、それに固定費3か月分の最低残高と遅れへの備えを足しました。現金残高は最も薄い年でも約${man(Math.round(PLAN.minCash / 100) * 100)}万円です。資金が豊富でも一度に使わず、線を越えたときだけ次を入れます。利用料は年額前払い、導入支援は着手金をいただき、黒字倒産を防ぎます。`],
  ["黒字化の条件は、拠点数より1拠点の台数。止まる線も先に決める",
    `では、どうすれば黒字になるか。同じ計算で確かめました。1拠点10台のままだと、計画どおりに拠点が増えても5年以内に黒字になりません。ところが、効果が出た班から台数を増やし、1拠点15台にできれば、新規が計画の6割でも5年目に黒字です。黒字の鍵は、拠点の数より、1拠点で使ってもらう台数です。そして、進むか止まるかの線を先に決めます。実証で1台1日${be}分以上減れば進む、15分未満なら撤退。1年目の終わりに試験導入3社のうち2社以上が本契約なら進む。2年目の終わりに平均12台に届かなければ採用を止め、3年目に継続率90%を越えていれば追加の資金を入れます。`],
  ["年ごとに、何をして、いくら売り、いくら残るか",
    `年単位の計画です。準備期は試作品と1拠点の実証。1年目は3社の試験導入で記録と報告の自動化、2年目は本契約に切り替えて${PR[1].sites}拠点、3年目は図面との照合を加え、大手1社を事例にして${PR[2].sites}拠点、4年目は表示付きグラスで手元ARを1工種から、5年目は教材化と電線工場への試験導入で${Y5.sites}拠点です。社員は${PR[0].staff}人から${Y5.staff}人、売上は約${mil(PR[0].sales)}百万円から約${mil(Y5.sales)}百万円。営業損益は5年間赤字で、5年目は約${mil(-Y5.op)}百万円の赤字です。黒字化には1拠点15台への拡大が条件です。`],
  ["10年で見れば、10台のままでも" + R10.firstBlack + "年目に黒字",
    `5年より先の10年でどうかを、同じ式で延ばしました。5年目までの計画は変えず、6年目からは新しい拠点を年36から40、社員は24人から32人と、採用のペースを落とします。すると、1拠点10台のままでも${R10.firstBlack}年目に黒字になり、${R10.cumBack}年目には5年間の赤字を取り戻します。10年目の営業利益は約${mil(R10.rows[9].op)}百万円、15台なら約${mil(R10_15.rows[9].op)}百万円です。積み上げ型なので、時間が味方になります。前提を一つずつ崩しても、どれも10年以内に黒字になります。ただし値下げとAI費用2倍では、現金がいったんマイナスになり、自己資金の追加が要ります。電線工場や海外への展開は入れていません。`],
  ["準備期は効果、導入期は売れるか、成長期は台数と人、成熟期は競合",
    "各段階で想定される課題です。準備期の課題は、効果が本当に出るか。普段使いのグラスは電池と録画の制約があり、記録が途切れるおそれがあります。だから要所だけを記録し、写真台帳から自動にして、2〜3班で小さく測ります。導入期の課題は、売れるか、定着するか。300万円を払っていただけるかはまだ確かめていません。現場が監視だと感じれば使われません。成長期の課題は、台数と人。10台のままでは黒字にならず、コンサル部隊とエンジニアの採用が追いつかないおそれがあります。成熟期の課題は、競合のAI化と、データの持ち主の取り決め、工場や海外への広げ方です。それぞれに判定の線を置いています。"],
  ["自分で見つけた弱いところを、確かめ方とセットで示す",
    `計画を自分で疑って、実現が危ういところを洗い出しました。一つ目は普段使いのグラスの性能です。一般向けのAIグラスは、動画が1回3分程度まで、開発者向けの映像は720pで、連続で撮ると電池が持ちません。二つ目は安全。保護めがねの規格に合うカメラ付きグラスはまだ少ない。三つ目はデータの通り道。一般向けのグラスはメーカーのアプリを経由します。四つ目は値段で、年300万円はまだ誰にも確かめておらず、得は払う額の約${CV.ratio.toFixed(1)}倍と余裕が小さい。五つ目は、10台のままでは黒字にならないこと。ほかに、採用、獲得費用に私の営業時間を入れていないこと、解約率の根拠です。どれも準備期と1年目に確かめる方法を決めています。`],
  ["準備期の6か月で、最初の1拠点を動かす",
    `準備期の6か月です。最初の課題、効果が出るかを確かめます。費用は約500万円。試作品の開発に250万円、グラスに80万円、1拠点での実証に100万円、設立と契約に40万円、予備30万円です。自己資金でまかない、私の給料はゼロです。3か月で試作品、6か月で実証の結果を出し、1台1日${be}分減ったか、グラスで1日記録できるかを確かめます。合格したら、1年目の2億円を入れて採用を始めます。`],
  ["最大のリスクは自分。だから役割と判断基準を先に決めておく",
    "最後にリスクです。最大のリスクは、私自身がボトルネックになること。だから役割を絞り、判断の基準を先に決めて、私がいなくても回る会社にします。進むかどうかは、思いではなく数字で決めます。そのうえで、私がこの事業を続けたい理由をお話しします。"],
  ["事業への思い。黒字は目的ではなく、続けるための条件",
    `最後に、私の願いです。10年続ければ、書類に使っていた約${man(Math.round(IMPACT.hours / 1e4))}万時間が、現場と若手の指導に戻ります。価値にして約${Math.round(IMPACT.value / 1e4)}億円です。私にとって、利益は目的ではなく、続けるための条件です。配当は求めず、利益は現場の支援と開発に戻します。自己資金は、見返りを求めない投資です。クラウドファンディングのように、共感してくださる工事会社や職人、業界団体と一緒に育てたい。記録から作った教材の一部は、業界と学校に無償で公開します。それでも資金は有限なので、止まる線は守ります。続けられる形を探し続けることが、私の責任です。日本の品質を支えてきた現場の勘を、人に依存しない形で次の担い手へ。私の資金と時間は、そのための種です。最初の一歩として、6か月で最初の1拠点を動かします。ご清聴ありがとうございました。`],
];

// 時間は台本の文字数から自動で出す（1秒あたり約8.5文字）
const mmss = (t) => `${Math.floor(t / 60)}:${String(t % 60).padStart(2, "0")}`;
const NOTES = {};
{
  let t = 0;
  NOTE_LIST.forEach((p, n) => {
    if (!p) return;
    const d = Math.max(15, Math.round(p[1].length / 8.5 / 5) * 5);
    NOTES[n] = `【${mmss(t)} - ${mmss(t + d)}】\n要約：${p[0]}\n\n${p[1]}`;
    t += d;
  });
  NOTES.total = mmss(t);
}
const ASSETS = path.join(__dirname, "assets");
const chap = (n, t) => `${n} ${t}`;


async function main() {
  const pres = new pptxgen();
  const add = await TH.setup(pres, "株式会社カンツギ 事業計画書", NOTES);
  const markBlack = await logo("kantsugi-cube-mono");
  const markWhite = await logo("kantsugi-cube-reverse");
  const wordWhite = await logo("kantsugi-logotype-reverse");
  const logoColor = await logo("kantsugi-logotype-color");

  // ---------- Slide 1 表紙 ----------
  {
    const s = add(1);
    s.addImage({ data: logoColor, x: 0.45, y: 0.38, w: 1.9, h: 1.9 * 120 / 560 });
    txt(s, "事業計画書（本編）", { x: 0.45, y: 0.9, w: 5.6, h: 0.26, size: 10.5, color: K.gray700 });
    txt(s, "現場の勘を次の担い手へ", { x: 0.45, y: 1.16, w: 5.6, h: 0.6, size: 26, bold: true, color: K.ink, valign: "middle" });
    txt(s, "人が減っても、現場が迷わず回る仕組みをつくる。AIグラスで熟練者の作業を記録し、書類づくりと若手の育成を自動化する", { x: 0.45, y: 1.8, w: 5.6, h: 0.5, size: 10.5 });
    hline(s, 0.45, 2.42, 9.1, K.ink, 1.25);
    box(s, { x: 0.45, y: 2.6, w: 5.6, h: 1.3, fill: TH.PALE, name: "first-step" });
    txt(s, "最初の目標", { x: 0.65, y: 2.68, w: 2, h: 0.3, size: 11, bold: true, color: K.gray700 });
    txt(s, "6か月で1拠点を動かす", { x: 0.65, y: 2.98, w: 5.2, h: 0.55, size: 24, bold: true, color: K.ink, valign: "middle" });
    txt(s, `3か月で試作品、3か月で実証。合格ラインは「1台1日${be}分の削減」`, { x: 0.65, y: 3.52, w: 5.2, h: 0.3, size: 10, color: K.gray700, valign: "middle" });
    txt(s, [run("社名：", { bold: true, color: K.ink }), run("カンツギ＝勘継。割れた器を金で継ぐ金継ぎのように、途切れそうな現場の勘をデータで継ぐ")], { x: 0.45, y: 4.1, w: 5.6, h: 0.5, size: 9.5, valign: "middle" });
    // 地中送電ケーブル（CV）の断面：外側から シース・遮へい層・外部半導電層・絶縁体・内部半導電層・導体
    const cx = 7.15, cy = 3.25, layers = [[0.66, "3A3F44"], [0.56, "8C6239"], [0.5, "2B2B2B"], [0.46, "F2F4F6"], [0.24, "2B2B2B"], [0.2, "B87333"]];
    layers.forEach(([r, c]) => s.addShape("ellipse", { x: cx - r, y: cy - r, w: 2 * r, h: 2 * r, fill: { color: c }, line: { color: "2B2B2B", width: 0.5 } }));
    [["シース", -0.42, -0.5], ["遮へい層", -0.53, -0.16], ["絶縁体", 0.3, 0.12], ["導体", 0.08, 0.38]].forEach(([n, dx, dy], i) => {
      const ly = 2.7 + i * 0.36, ex = cx + dx, ey = cy + dy;
      s.addShape("line", { x: Math.min(ex, 8.05), y: Math.min(ey, ly + 0.11), w: Math.abs(8.05 - ex), h: Math.abs(ly + 0.11 - ey), flipV: ly + 0.11 < ey && ex < 8.05 ? true : false, line: { color: K.gray700, width: 0.5 } });
      txt(s, n, { x: 8.1, y: ly, w: 1.4, h: 0.22, size: 8.5, color: K.gray700, valign: "middle" });
    });
    txt(s, "地中送電ケーブル（CV）の断面", { x: 6.35, y: 4.0, w: 3.2, h: 0.22, size: 8, color: K.gray700, align: "center" });
    box(s, { x: 6.6, y: 5.2, w: 3.28, h: 0.305, line: K.ink });
    txt(s, [run("件名 ", { color: K.gray700 }), run("株式会社カンツギ 事業計画書", { bold: true, color: K.ink })], { x: 6.68, y: 5.2, w: 2.6, h: 0.305, size: 8, valign: "middle" });
    txt(s, "1／33", { x: 9.3, y: 5.2, w: 0.58, h: 0.305, size: 9, bold: true, color: K.ink, align: "center", valign: "middle" });
    s.addShape("line", { x: 9.3, y: 5.2, w: 0, h: 0.305, line: { color: K.ink, width: 0.75 } });
  }

  // ---------- Slide 設問との対応 ----------
  {
    const s = add(2);
    heading(s, "発表の構成 ─ 課題の設問①〜⑤の順に話す", "左上の番号が設問の番号。計算の過程は別冊「計算詳細版」（右上にページ）");
    table(s, [["設問", "問い", "本編", "一言の答え"],
      [{ text: "①", bold: true, color: ACC }, "どんな事業か・選んだ理由・想い", "p.3〜15", "AIグラスで現場の作業を記録し、報告と育成を自動にする。施工管理で見た「勘が消える」を止めたい"],
      [{ text: "②", bold: true, color: ACC }, "どんなスタッフと働くか", "p.16", `5年目${Y5.staff}人。現場監修と導入コンサルタント（コンサル部隊）を社内に持つ。私は型と判断の基準をつくる`],
      [{ text: "③", bold: true, color: ACC }, "経営理念（ホームページで伝えること）", "p.17", "現場の勘を次の担い手へ。監視ではなく、技術を残す仕組み"],
      [{ text: "④", bold: true, color: ACC }, "年単位の経営計画と収支", "p.18〜28", `1拠点10台で年300万円。5年では赤字、10年なら${R10.firstBlack}年目に黒字。15台なら5年目に黒字`],
      [{ text: "⑤", bold: true, color: ACC }, "各段階で想定される課題", "p.29〜33", "準備期は効果、導入期は売れるか、成長期は台数と人、成熟期は競合。最後に創業者の願い"]],
      { x: X0, y: 1.15, w: W, colW: [0.55, 2.55, 0.85, 5.25], rowH: 0.5, size: 9, align: "left", name: "toc" });
    box(s, { x: X0, y: 4.25, w: W, h: 0.62, fill: K.ink, name: "premise" });
    txt(s, [run("課題の前提：", { bold: true, color: L.YELLOW }), run("来年、人生の転機があり、豊富な資金をもとに新しい事業を立ち上げて経営する。", { breakLine: true }),
      run("資金は自己資金でまかなう（融資・出資は使わない）。数字はすべて一つの計算モデルから出している")], { x: X0 + 0.15, y: 4.25, w: W - 0.3, h: 0.62, size: 9.5, color: K.paper, valign: "middle" });
  }

  // ---------- Slide 2 ①事業概要 ----------
  {
    const s = add(3);
    heading(s, chap("①", "事業概要 ─ 現場の作業を記録し、報告と教育を自動に"), "AIグラスを掛けて作業するだけ。報告書・図面との照合・教材までをまとめて届ける");
    const q = [
      ["何のビジネス？", "職長や施工管理者がAIグラスを掛けるだけで、一人称の動画と会話を記録。AIが工事写真を整理して報告書を作り、図面と照合し、教材にするサービス"],
      ["既存の市場との違い", "施工管理アプリは人が撮って入力する器。当社は自動で記録し「照合→報告→教材」までまとめて提供。施工経験者が中身を監修する"],
      ["誰のどんな悩みを解決する？", "関東の中堅電気工事会社。残業の上限規制と高齢化で、工事写真・報告書と若手の育成に回す時間がない。熟練者が辞めると判断が消える"],
      ["収益構造", "会社の基盤 月5万円＋端末1台 月2万円の利用料（標準10台で年300万円）が毎年積み上がる。初年度だけ導入支援200万円"],
    ];
    const cw = (W - 0.15) / 2, ch = 1.62, top = 1.15;
    q.forEach(([h, v], i) => {
      const x = X0 + (i % 2) * (cw + 0.15), y = top + Math.floor(i / 2) * (ch + 0.15);
      box(s, { x, y, w: cw, h: ch, fill: i === 3 ? K.ink : K.fill, name: "summary-" + i });
      const c = i === 3 ? K.paper : K.ink;
      txt(s, h, { x: x + 0.18, y: y + 0.12, w: cw - 0.36, h: 0.34, size: 13, bold: true, color: c });
      txt(s, v, { x: x + 0.18, y: y + 0.52, w: cw - 0.36, h: ch - 0.62, size: 11, color: c });
    });
    txt(s, "最初の顧客：関東の中堅電気工事会社（施工管理者30人以上）。成長期以降に電線・素材の工場へ広げる", { x: X0, y: 4.75, w: W, h: 0.3, size: 10, color: K.gray700 });
  }

  // ---------- Slide ①選んだ理由と想い ----------
  {
    const s = add(4);
    heading(s, chap("①", "選んだ理由と想い ─ 現場の勘が消えるのを見てきた"), "お金を払う理由がある市場、使える技術、そして自分がやる理由。3つがそろった");
    const top = 1.15, cw = 2.97, gap = 0.145, ch = 2.15;
    [
      ["市場の理由", "払う理由がある", ["建設業の55歳以上は36.7%、29歳以下は11.7%", "2024年から残業に上限。書類と育成に回す人がいない", "手戻りだけで建設全体に年約1.5兆円"]],
      ["技術の理由", "今なら作れる", ["見て理解するAI（画像・動画のAI）が実用の水準に", "カメラ付きAIグラスが普段のメガネと同じ約50gに", "表示付きのグラスも発売（2025年）"]],
      ["自分の理由", "私がやる", ["電線メーカーで地中送電線の施工管理を5年", "ベテランの判断が手順書に残らず、退職で消えるのを見た", "AIの仕組みを独学で作り、毎日動かしている"]],
    ].forEach(([h, sub, items], i) => {
      const x = X0 + i * (cw + gap);
      box(s, { x, y: top, w: cw, h: ch, line: K.rule, name: "reason-" + i });
      box(s, { x, y: top, w: cw, h: 0.55, fill: i === 2 ? ACC : K.ink });
      txt(s, [run(h, { bold: true, fontSize: 12.5, breakLine: true }), run(sub, { fontSize: 9 })], { x: x + 0.14, y: top + 0.02, w: cw - 0.28, h: 0.52, color: K.paper, valign: "middle" });
      txt(s, items.map((v, j) => run("・" + v, { breakLine: j < items.length - 1 })), { x: x + 0.14, y: top + 0.65, w: cw - 0.28, h: ch - 0.75, size: 9.5, psa: 5 });
    });
    box(s, { x: X0, y: 3.45, w: W, h: 1.4, fill: K.ink, name: "omoi" });
    txt(s, [run("想い", { bold: true, color: L.YELLOW, fontSize: 11, breakLine: true }),
      run("割れた器を金で継ぐ「金継ぎ」のように、途切れそうな現場の勘をデータで継ぐ。", { bold: true, fontSize: 14, breakLine: true }),
      run("人が減っても、日本の現場の品質を落とさない。若手が「聞ける先輩がいない」と迷わない現場をつくる。それが社名「カンツギ（勘継）」の意味", { fontSize: 10 })],
      { x: X0 + 0.2, y: 3.45, w: W - 0.4, h: 1.4, color: K.paper, valign: "middle", psa: 4 });
    txt(s, "グラスの重さはRay-Ban Meta（第2世代）の公表値で約50g。年齢構成は国土交通省（2024年）", { x: X0, y: 4.98, w: W, h: 0.26, size: S.note, color: K.gray500 });
  }

  // ---------- Slide ①背景（なぜ今） ----------
  {
    const s = add(5);
    heading(s, chap("①", "背景 ─ AIは「頭」の支援から、作業者の「目と手」の支援へ"), "知的労働はAIで進んだ。次は肉体労働の暗黙知。人の目は口ほどに物を言う ─ 次は「見るAI」");
    const top = 1.15, cw = 2.75;
    const cols = [
      ["これまで", K.fill, K.ink, [["誰を", "管理者"], ["何を", "報告書・議事録などの事務作業"], ["AI", "考える・書くAI"], ["道具", "PC・スマホ"]]],
      ["これから", K.ink, K.paper, [["誰を", "作業者（ブルーカラー）"], ["何を", "安全・品質・施工手順の作業支援"], ["AI", "見るAI（視覚で現場を理解する）"], ["道具", "ARグラス・スマートグラス"]]],
    ];
    cols.forEach(([h, fill, c, rows], i) => {
      const x = X0 + i * (cw + 0.55);
      box(s, { x, y: top, w: cw, h: 3.1, fill, name: "era-" + i });
      txt(s, h, { x: x + 0.15, y: top + 0.1, w: cw - 0.3, h: 0.35, size: 14, bold: true, color: i === 1 ? L.YELLOW : ACC });
      rows.forEach(([k, v], j) => {
        const y = top + 0.55 + j * 0.62;
        txt(s, [run(k, { fontSize: 8.5, breakLine: true }), run(v, { bold: true, fontSize: 11 })], { x: x + 0.15, y, w: cw - 0.3, h: 0.58, color: c });
      });
    });
    arrow(s, X0 + cw + 0.08, top + 1.55, 0.4, 0, { lw: 3, color: ACC });
    const rx = 6.7, rw = 2.9;
    txt(s, "なぜ今か", { x: rx, y: top, w: rw, h: 0.28, size: 11, bold: true });
    [
      ["AIの判断力", "考える・書くAIは実用の水準。次は現場を「見て」理解する番"],
      ["端末", "表示付きのスマートグラスが発売（2025年）。ARグラスは2027年の一般発売が報道"],
      ["人手", "2024年から残業に上限。55歳以上が36.7%"],
      ["世代", "スマホ世代の次に、ARグラス世代が現場に入ってくる"],
    ].forEach(([k, v], i) => {
      const y = top + 0.34 + i * 0.68;
      hline(s, rx, y, rw, K.rule, T.stroke.rule);
      txt(s, [run(k, { bold: true, color: ACC, breakLine: true }), run(v)], { x: rx, y: y + 0.03, w: rw, h: 0.64, size: 9 });
    });
    box(s, { x: X0, y: 4.4, w: W, h: 0.5, fill: K.fill });
    txt(s, [run("目指す姿：", { bold: true, color: ACC }), run("作業者が抜けても、管理者が抜けても、現場を解釈できるAI（データ資産）が残る")], { x: X0 + 0.15, y: 4.4, w: W - 0.3, h: 0.5, size: 11, bold: true, valign: "middle" });
    txt(s, "出典：Meta Ray-Ban Display（2025年9月発売）、ARグラスの一般発売時期は報道（Bloomberg）で公式発表ではない。年齢構成は国土交通省（2024年）", { x: X0, y: 4.98, w: W, h: 0.26, size: S.note, color: K.gray500 });
  }

  // ---------- Slide 3 ②課題（ペイン） ----------
  {
    const s = add(6);
    heading(s, chap("①", "課題 ─ 報告と育成に追われ、熟練者が辞めると止まる"), "ターゲットは三者。それぞれに不満・不便・損失がある");
    const ps = [
      ["DX推進担当者", "現場で孤軍奮闘（推進する人）", ["ツールを入れても現場に定着しない", "成果を数字で説明できず、予算が続かない", "現場とシステムの言葉をつなぐ人がいない"]],
      ["施工管理者・電工", "人手不足の現場（使う人）", ["工事写真の整理・日報・報告書に毎日追われる", "聞ける熟練者がいない。手順書に勘どころが書いていない", "ミスに気づくのが後になり、手戻りが出る"]],
      ["工事部長・社長", "引退と後継者不足（お金を出す人）", ["2024年から残業に上限。人を増やせない", "熟練者の判断が退職とともに消える", "立会に行けず、現場の状況が見えない"]],
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
    [["36.7%", "建設業で働く人のうち55歳以上（2024年）(1)"], ["11.7%", "同じく29歳以下。次の担い手が足りない (1)"], ["月45時間", "2024年4月から建設業にも残業の上限（原則）(2)"], ["約1,940時間", "建設業の1年の総労働時間（残業込み・令和6年）(3)"]].forEach(([f, l], i) => {
      const cw2 = (W - 0.3) / 4, x = X0 + i * (cw2 + 0.1);
      box(s, { x, y: sy, w: cw2, h: 1.2, fill: K.fill, name: "pain-stat-" + i });
      txt(s, f, { x: x + 0.1, y: sy + 0.08, w: cw2 - 0.2, h: 0.45, size: 20, bold: true, valign: "middle" });
      txt(s, l, { x: x + 0.1, y: sy + 0.55, w: cw2 - 0.2, h: 0.6, size: 8.5 });
    });
    txt(s, "(1) 国土交通省（総務省「労働力調査」2024年平均から算出。建設業就業者477万人）　(2) 労働基準法の時間外労働の上限規制（建設業は2024年4月1日から適用）　(3) 毎月勤労統計調査 月161.5時間×12", { x: X0, y: 4.98, w: W, h: 0.26, size: S.note, color: K.gray500 });
  }

  // ---------- Slide ②損失の大きさ ----------
  {
    const s = add(7);
    detailRef(s, "7");
    heading(s, chap("①", "損失の大きさ ─ 手戻りだけで、建設全体に年1.5兆円"), "勘の喪失は思いの問題ではなく、お金の問題。熟練者が抜けるほど、やり直しと断る工事が増える");
    const top = 1.15, lw = 4.35;
    const big = [
      [`約${(NV.reworkAll / 1e8).toFixed(1)}兆円／年`, "建設全体の手戻り", "建設投資 約75.6兆円（2025年度見通し）× 手戻り2%"],
      [`約${man(Math.round(NV.reworkElectrical / 1e6) * 100)}億円／年`, "うち電気工事", "電気工事業の完成工事高 約12兆円 × 手戻り2%"],
      [`約${Math.round(NV.retiring / 10000)}万人`, "55歳以上の建設業就業者", "477万人 × 36.7%。10〜15年で引退に向かう"],
    ];
    big.forEach(([n, a, f], i) => {
      const y = top + i * 1.05;
      box(s, { x: X0, y, w: lw, h: 0.95, fill: i === 0 ? K.ink : K.fill, name: "loss-" + i });
      const c = i === 0 ? K.paper : K.ink;
      txt(s, n, { x: X0 + 0.15, y: y + 0.06, w: 2.55, h: 0.5, size: 17, bold: true, color: i === 0 ? L.YELLOW : ACC, valign: "middle" });
      txt(s, a, { x: X0 + 2.7, y: y + 0.1, w: lw - 2.8, h: 0.42, size: 9.5, bold: true, color: c, valign: "middle" });
      txt(s, f, { x: X0 + 0.15, y: y + 0.58, w: lw - 0.3, h: 0.32, size: 8.5, color: i === 0 ? K.paper : K.gray700, valign: "middle" });
    });
    const rx = 5.1, rw = 4.5;
    txt(s, "勘が消えると、なぜお金が消えるか", { x: rx, y: top, w: rw, h: 0.28, size: 11, bold: true });
    const chain = [["熟練者が辞める", "判断の理由は手順書にない"], ["若手が迷う・間違える", "寸法・基準点・重なり順の取り違い"], ["手戻りと工期遅れ", "材料・人件費・やり直しの費用"], ["残業の上限で人を足せない", "受けられる工事が減る"]];
    chain.forEach(([h, d], i) => {
      const y = top + 0.36 + i * 0.66;
      box(s, { x: rx, y, w: rw, h: 0.52, fill: i === 3 ? ACC : K.fill });
      txt(s, [run(h, { bold: true }), run("　" + d, { fontSize: 8.5 })], { x: rx + 0.12, y, w: rw - 0.24, h: 0.52, size: 10, color: i === 3 ? K.paper : K.ink, valign: "middle" });
      if (i < 3) arrow(s, rx + rw / 2, y + 0.53, 0, 0.12, { lw: 1.5 });
    });
    box(s, { x: X0, y: 4.35, w: W, h: 0.55, line: ACC, lw: 1.5 });
    txt(s, [run("当社が5年目に届ける価値：", { bold: true, color: ACC }), run(`${Y5.sites}拠点 × 年約${man(CV.docs)}万円（書類の時間だけ）＝ 年約${(NV.somValue / 1e4).toFixed(1)}億円。手戻りの削減が実測できれば上乗せ`)], { x: X0 + 0.15, y: 4.35, w: W - 0.3, h: 0.55, size: 10.5, valign: "middle" });
    txt(s, "手戻り率2%は控えめな仮定（海外研究で工事費の0.4〜6%）。建設投資は国交省の見通し、完成工事高は施工統計の二次集計", { x: X0, y: 4.98, w: W, h: 0.26, size: S.note, color: K.gray500 });
  }

  // ---------- Slide 4 ③解決策・他社がすぐに真似できない理由 ----------
  {
    const s = add(8);
    heading(s, chap("①", "解決策 ─ 記録は自動。貯まるほど他社は真似しにくい"), "一人称の動画と会話をAIが整理し、若手・管理者・次の担い手に渡す。その先にフィジカルAI");
    s.addImage({ path: path.join(ASSETS, "shot-overview.png"), x: X0, y: 1.15, w: 5.6, h: 5.6 / 2.375 });
    txt(s, "サービス内容", { x: X0, y: 3.62, w: 5.6, h: 0.26, size: 10.5, bold: true });
    txt(s, [
      run("若手へ：", { bold: true }), run("図面・作業手順を現物に重ね、差異をその場で表示", { breakLine: true }),
      run("管理者へ：", { bold: true }), run("報告書の手順に沿って写真を抽出し、当日の報告書を自動作成", { breakLine: true }),
      run("次の担い手へ：", { bold: true }), run("勘どころを解説動画・音声番組・研修資料にして当社が届ける"),
    ], { x: X0, y: 3.9, w: 5.6, h: 0.95, size: 9.5, psa: 3 });
    const rx = 6.2, rw = 3.4;
    box(s, { x: rx, y: 1.15, w: rw, h: 3.7, fill: K.ink, name: "moat" });
    txt(s, "他社がすぐに真似できない3つの理由", { x: rx + 0.15, y: 1.22, w: rw - 0.3, h: 0.3, size: 11.5, bold: true, color: K.paper });
    [
      ["記録が貯まるほど強くなる", "図面に現れない現場処置の記録は使うほど増え、後発には移せない"],
      ["意味づけは現場経験者にしかできない", "映像のどこが重要かを見分けて意味を付けるのは、施工の経験者"],
      ["業界の型を先に揃える", "電気工事の工種ごとの手順と報告書の型を、業界で最初に揃える"],
    ].forEach(([h, d], i) => {
      const y = 1.62 + i * 1.05;
      txt(s, [run((i + 1) + ". " + h, { bold: true, breakLine: true, fontSize: 10.5 }), run(d, { fontSize: 9 })], { x: rx + 0.15, y, w: rw - 0.3, h: 0.98, color: K.paper, psa: 2 });
    });
    txt(s, "ノウハウの核：判断ラベル付きの現場データ（どの手順の、どの場面が、なぜ重要か）", { x: X0, y: 4.98, w: W, h: 0.26, size: S.note, color: K.gray500 });
  }

  // ---------- Slide ③導入効果 ----------
  {
    const s = add(9);
    detailRef(s, DP.value);
    heading(s, `① 導入効果 ─ 1拠点10台で年約${man(Math.round(CV.docs / 10) * 10)}万円。支払いの約${CV.ratio.toFixed(1)}倍`, "得をするのは、AIグラスを掛けた班だけ。確かな部分（書類と写真整理の時間）だけを計画に入れる");
    const top = 1.15, lw = 5.4;
    box(s, { x: X0, y: top, w: lw, h: 1.05, fill: K.fill });
    txt(s, [run("式：", { bold: true, color: ACC }), run(`10台 × 年240日 ×（職長15分×${man(Math.round(V.H.electrician * 100) * 100)}円 ＋ 施工管理者20分×${man(Math.round(V.H.manager * 100) * 100)}円）`, { breakLine: true }),
      run(`＝ 年 約${man(CV.docs)}万円`, { bold: true, fontSize: 18, color: ACC })], { x: X0 + 0.15, y: top, w: lw - 0.3, h: 1.05, size: 10, valign: "middle", psa: 3 });
    txt(s, `時給＝統計の年収×会社負担${V.WAGE.employerLoad}÷残業を含む年${man(V.WAGE.annualHours)}時間。1台1日35分は仮説で、実証で測る`, { x: X0, y: top + 1.1, w: lw, h: 0.28, size: 8, color: K.gray700 });
    const bars = [["お客様の効果（書類）", CV.docs, ACC], ["お客様の支払い", CV.cost, K.ink], ["手戻りも減れば（上振れ）", CV.docs + CV.rework, K.gray500]];
    const sc = (lw - 2.95) / (CV.docs + CV.rework);
    bars.forEach(([n, v, c], i) => {
      const y = top + 1.5 + i * 0.5;
      txt(s, n, { x: X0, y, w: 1.9, h: 0.38, size: 9.5, bold: true, valign: "middle" });
      box(s, { x: X0 + 1.95, y: y + 0.06, w: v * sc, h: 0.26, fill: c });
      txt(s, man(v) + "万円", { x: X0 + 2.0 + v * sc, y, w: 0.95, h: 0.38, size: 10, bold: true, valign: "middle" });
    });
    txt(s, `支払い＝当社への利用料300万円＋グラスのリース約${man(V.leaseSite())}万円`, { x: X0, y: top + 3.0, w: lw, h: 0.25, size: 8, color: K.gray700 });
    const rx = 6.1, rw = 3.5;
    box(s, { x: rx, y: top, w: rw, h: 1.5, fill: K.ink });
    txt(s, [run("元が取れる条件", { bold: true, color: L.YELLOW, breakLine: true }), run(`1台1日 ${Math.round(CV.breakEvenMin)}分`, { bold: true, fontSize: 20, breakLine: true }), run("想定は35分。実証の合格ラインにする", { fontSize: 9 })], { x: rx + 0.15, y: top, w: rw - 0.3, h: 1.5, size: 10, color: K.paper, valign: "middle", psa: 3 });
    box(s, { x: rx, y: top + 1.62, w: rw, h: 1.63, fill: K.fill });
    txt(s, [run("計画に入れない2つ", { bold: true, breakLine: true }), run(`手戻り：年${man(CV.rework)}万円（30億円×2%×対象工種30%×削減20%）。実測できれば約${CV.ratioUp.toFixed(1)}倍`, { fontSize: 9, breakLine: true }),
      run("新人の指導：金額にせず、独り立ちまでの月数で測る", { fontSize: 9 })], { x: rx + 0.15, y: top + 1.62, w: rw - 0.3, h: 1.63, size: 10, valign: "middle", psa: 4 });
    box(s, { x: X0, y: 4.5, w: W, h: 0.42, line: ACC, lw: 1.5 });
    txt(s, [run("言い方：", { bold: true, color: ACC }), run(`「1台1日${Math.round(CV.breakEvenMin)}分減れば元が取れます。想定は35分。手戻りも減れば約${CV.ratioUp.toFixed(1)}倍です」`)], { x: X0 + 0.15, y: 4.5, w: W - 0.3, h: 0.42, size: 10, valign: "middle" });
  }

  // ---------- Slide ③手元AR ----------
  {
    const s = add(10);
    heading(s, "① 手元AR ─ 壁の図面を、手元に実物大で出す", "照合ができなくても価値がある。長く層の多いケーブルジョイントは、ARの効果が最も出やすい");
    const top = 1.12, lw = 4.3;
    // before
    box(s, { x: X0, y: top, w: lw, h: 0.8, fill: K.fill });
    txt(s, [run("今：", { bold: true, color: ACC }), run("約2mのジョイントの組立図を、わら半紙で壁に貼る。壁を見る → 手元に戻る → 寸法を思い出す → 測る、の往復")], { x: X0 + 0.12, y: top + 0.03, w: lw - 0.24, h: 0.74, size: 9, valign: "middle" });
    // after: cable with hologram overlay
    const cy = top + 1.3, cx = X0 + 0.2, len = lw - 0.4;
    txt(s, [run("ARで：", { bold: true, color: ACC }), run("実物大・立体・工程ごとの完成形が手元に")], { x: X0, y: top + 0.86, w: lw, h: 0.24, size: 9 });
    const segs = [[0.0, 0.18, K.gray700], [0.18, 0.34, K.gray500], [0.34, 0.66, K.rule], [0.66, 0.82, K.gray500], [0.82, 1.0, K.gray700]];
    segs.forEach(([a, b, c]) => box(s, { x: cx + a * len, y: cy + 0.18, w: (b - a) * len, h: 0.26, fill: c }));
    s.addShape("rect", { x: cx + 0.34 * len, y: cy - 0.02, w: 0.32 * len, h: 0.66, fill: { color: K.paper, transparency: 100 }, line: { color: ACC, width: 1.5, dashType: "dash" } });
    txt(s, "ホログラム（工程3の完成形）", { x: cx + 0.2 * len, y: cy - 0.24, w: 0.6 * len, h: 0.22, size: 8, color: ACC, bold: true, align: "center" });
    hline(s, cx + 0.34 * len, cy + 0.72, 0.32 * len, ACC, 1);
    txt(s, "剥ぎ取り 120mm", { x: cx + 0.34 * len, y: cy + 0.74, w: 0.32 * len, h: 0.2, size: 7.5, color: ACC, align: "center" });
    // red / blue: mistake-prone point
    const ry = top + 2.18;
    box(s, { x: X0, y: ry, w: lw, h: 0.92, line: K.ink, lw: 1.25, name: "red-blue" });
    s.addShape("line", { x: X0 + 0.2, y: ry + 0.25, w: 1.0, h: 0, line: { color: L.RED, width: 4 } });
    s.addShape("line", { x: X0 + 0.2, y: ry + 0.45, w: 1.0, h: 0, line: { color: "2F6DB5", width: 4 } });
    txt(s, "✂", { x: X0 + 1.22, y: ry + 0.12, w: 0.3, h: 0.3, size: 14, color: K.ink });
    txt(s, "赤を切れ", { x: X0 + 0.2, y: ry + 0.56, w: 1.3, h: 0.3, size: 10, bold: true, color: L.RED });
    txt(s, [run("ミスしやすい所で止める", { bold: true, breakLine: true }), run("過去のミス事例を学習させ、現物を認識したら音声で警告し、レーザーポインターで正しい位置を示す（赤か青か、の瞬間を外さない）", { fontSize: 8.5 })], { x: X0 + 1.6, y: ry + 0.04, w: lw - 1.7, h: 0.84, size: 9.5, valign: "middle" });
    // benefits
    const rx = 4.95, rw = 4.65;
    txt(s, "強調したい6つの効果", { x: rx, y: top, w: rw, h: 0.26, size: 10.5, bold: true });
    [
      ["視線と体の移動をなくす", "図面を見に行く往復が消え、記憶違いと中断が減る。手袋のまま声で操作"],
      ["実物大（1:1）で比べる", "縮小図を頭で実寸に直さない。基準点の取り違いに気づける"],
      ["層の重なりを立体・工程順に", "断面・半透明で、今どこまで進み次はどの形かが見える"],
      ["ベテランの注意点を固定表示", "「ここは傷をつけない」を該当箇所に。新人の立ち上がりが早い"],
      ["常に最新版・品質の均一化", "古い版・汚れ・紛失がない。誰でも同じ基準を見る"],
      ["記録と事前トレーニング", "一人称映像を施工記録に。同じデータで事務所で模擬施工"],
    ].forEach(([h, d], i) => {
      const y = top + 0.32 + i * 0.47;
      hline(s, rx, y, rw, K.rule, T.stroke.rule);
      txt(s, [run((i + 1) + "  " + h, { bold: true, breakLine: true }), run(d, { fontSize: 8.5, color: K.gray700 })], { x: rx, y: y + 0.02, w: rw, h: 0.45, size: 9.5 });
    });
    box(s, { x: X0, y: 4.28, w: W, h: 0.62, fill: K.ink });
    txt(s, [
      run("副次効果：", { bold: true, color: L.YELLOW }), run("異物・溶接の火花から目を守る（保護めがねの規格適合は要確認）　", {}),
      run("限界：", { bold: true, color: L.YELLOW }), run("位置合わせには数mm〜cmのずれ。寸法の最終確認はスケールで。役割は「測る道具」でなく「理解して間違いに気づく道具」"),
    ], { x: X0 + 0.15, y: 4.28, w: W - 0.3, h: 0.62, size: 8.5, color: K.paper, valign: "middle" });
    txt(s, "効果の測り方：手戻り・施工ミスの件数、1ジョイントの作業時間、新人が独り立ちするまでの期間。まずよく使うジョイント1種類で実証する", { x: X0, y: 4.98, w: W, h: 0.26, size: S.note, color: K.gray500 });
  }

  // ---------- Slide ③育つAI（学習ループ） ----------
  {
    const s = add(11);
    heading(s, "① 育つAI ─ ベテランが解釈を教え、使うほど賢くなる", "撮る → 教える → 切り取る → 集める → 広げる。ループが回るほど、現場を解釈できるAIが育つ");
    const cx = 2.65, cy = 2.8, RX = 1.6, RY = 1.2;
    s.addShape("ellipse", { x: cx - RX, y: cy - RY, w: 2 * RX, h: 2 * RY, fill: { color: K.paper }, line: { color: K.rule, width: 1.5, dashType: "dash" } });
    box(s, { x: cx - 0.68, y: cy - 0.38, w: 1.36, h: 0.76, fill: K.ink });
    txt(s, [run("現場を解釈できるAI", { bold: true, breakLine: true }), run("＝データ資産", { color: L.YELLOW, bold: true })], { x: cx - 0.68, y: cy - 0.38, w: 1.36, h: 0.76, size: 9, color: K.paper, align: "center", valign: "middle" });
    const nodes = [["撮る", "一人称で常時記録"], ["教える", "ベテランが口頭で\n「この場面が大事」"], ["切り取る", "AIが必要な場面を\n抜き出して報告"], ["集める", "意味のあるデータに\n絞って収集"], ["広げる", "別の作業・現場へ"]];
    nodes.forEach(([h, d], i) => {
      const ang = -Math.PI / 2 + (i * 2 * Math.PI) / 5;
      const x = cx + RX * Math.cos(ang), y = cy + RY * Math.sin(ang);
      box(s, { x: x - 0.66, y: y - 0.32, w: 1.32, h: 0.64, fill: i === 1 ? ACC : K.fill, line: i === 1 ? undefined : K.rule });
      txt(s, [run(h, { bold: true, fontSize: 10.5, breakLine: true }), run(d, { fontSize: 7.5 })], { x: x - 0.66, y: y - 0.32, w: 1.32, h: 0.64, color: i === 1 ? K.paper : K.ink, align: "center", valign: "middle", lsm: 1.0 });
    });
    const rx = 5.4, rw = 4.2;
    txt(s, "育ったAIが現場でやること", { x: rx, y: 1.15, w: rw, h: 0.26, size: 10.5, bold: true });
    [
      ["報連相の高速化", "口頭の報告・確認が、撮った瞬間に自動で届く"],
      ["安全の声かけ", "危険箇所の注意、「30分作業したので休憩しましょう」"],
      ["品質の見張り", "過去のミス事例の場面で警告（手元AR）"],
      ["VR施工シミュレーター", "集めた場面で、事務所で模擬施工"],
      ["教材", "勘どころを動画・音声・研修資料に"],
    ].forEach(([h, d], i) => {
      const y = 1.47 + i * 0.5;
      hline(s, rx, y, rw, K.rule, T.stroke.rule);
      txt(s, [run(h, { bold: true, breakLine: true }), run(d, { fontSize: 8.5, color: K.gray700 })], { x: rx, y: y + 0.02, w: rw, h: 0.48, size: 9.5 });
    });
    box(s, { x: X0, y: 4.35, w: W, h: 0.58, fill: K.ink });
    txt(s, [run("暗黙知が形式知へ：", { bold: true, color: L.YELLOW }), run("情報を、楽に・大量に・抜け目なく・再現性をもって・正確に・速く伝える。人が抜けても、解釈できるAIが残る")], { x: X0 + 0.15, y: 4.35, w: W - 0.3, h: 0.58, size: 10, color: K.paper, valign: "middle" });
    txt(s, "「教える」はベテランの管理者の口頭指示と、場面へのタグ付け。真似されにくさの源泉（意味づけは施工経験者にしかできない）", { x: X0, y: 4.98, w: W, h: 0.26, size: S.note, color: K.gray500 });
  }

  // ---------- Slide 5 デモ ----------
  {
    const s = add(12);
    heading(s, "① デモ ─ 図面との差異を見つけ、報告書が自動でできる", "架空のケーブル接続工事。若手の作業を記録し、8mmの差異を検出、報告書と教材までつながる");
    txt(s, "① 作業者のARグラス", { x: X0, y: 1.1, w: 5.4, h: 0.24, size: 10, bold: true });
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

  // ---------- Slide ①端末と使い方 ----------
  {
    const s = add(13);
    detailRef(s, DP.device);
    heading(s, chap("①", "端末と使い方 ─ メガネと同じ軽さのAIグラスを職長が掛ける"), "グラスは作らず市販品を選ぶ。誰が・どこで・何台・何を撮るかを先に決める");
    const top = 1.15, lw = 4.6;
    txt(s, "使い方（1拠点の例）", { x: X0, y: top, w: lw, h: 0.26, size: 10.5, bold: true });
    table(s, [["", "決めごと"],
      ["誰が", "1班に1台、職長が掛ける（電工全員には配らない）"],
      ["何台", "1拠点10台＝5現場×2班。効果が出た班から増やす"],
      ["どこで", "作業エリアだけ。休憩・移動・私語は記録しない"],
      ["何を撮る", "作業の区切り・声の合図・検査の場面を短く切り出す。録画は端末に残し、AIに送るのは選んだ場面だけ"],
      ["どう送る", "グラス → スマホ → クラウド。圏外は後で送る"],
      ["誰が見る", "施工管理者は報告書を確認、熟練者は判断を足す、若手は教材で学ぶ"]],
      { x: X0, y: top + 0.3, w: lw, colW: [0.85, 3.75], rowH: [0.26, 0.3, 0.3, 0.3, 0.5, 0.3, 0.42], size: 8.5, leftCols: [1], name: "usage" });
    const rx = 5.25, rw = 4.35;
    txt(s, "端末の構成（標準10台）", { x: rx, y: top, w: rw, h: 0.26, size: 10.5, bold: true });
    [
      ["8台", "カメラ付きAIグラス", "約8万円・約50g。記録と音声メモ（例：Ray-Ban Meta 第2世代）", ACC],
      ["2台", "表示付きARグラス", "約12万円・約70g。手元に図面と手順（例：Meta Ray-Ban Display）", K.ink],
      ["選択", "産業用ヘッドセット", "約37万円〜。ヘルメット装着・防じん。粉じん・高所の現場だけ（例：RealWear）", K.gray500],
    ].forEach(([n, a, b, c], i) => {
      const y = top + 0.32 + i * 0.66;
      box(s, { x: rx, y, w: 0.62, h: 0.58, fill: c });
      txt(s, n, { x: rx, y, w: 0.62, h: 0.58, size: 11, bold: true, color: K.paper, align: "center", valign: "middle" });
      txt(s, [run(a, { bold: true, breakLine: true }), run(b, { fontSize: 8.5, color: K.gray700 })], { x: rx + 0.7, y, w: rw - 0.7, h: 0.58, size: 9.5, valign: "middle" });
    });
    box(s, { x: X0, y: 3.95, w: W, h: 0.92, line: ACC, lw: 1.5 });
    txt(s, [
      run("当社が作るもの：", { bold: true, color: ACC }), run("アプリ・AI・データ基盤（端末をまたいで同じデータを使う）。グラス本体は作らず、進化したら乗り換える", { breakLine: true }),
      run("グラスの持ち方：", { bold: true, color: ACC }), run(`お客様がリース（10台で年約${man(V.leaseSite())}万円）。試験導入はレンタル、3年目からは端末込みの料金も用意（詳細版 p.29）`),
    ], { x: X0 + 0.15, y: 3.95, w: W - 0.3, h: 0.92, size: 9, valign: "middle", psa: 3 });
    txt(s, "価格・重さは公表値と販売店の参考価格。国内の正規販売・技適・保護めがねの規格・電池の持ちは準備期に実機で確認する（詳細版 p.26）", { x: X0, y: 4.98, w: W, h: 0.26, size: S.note, color: K.gray500 });
  }

  // ---------- Slide ①競合 ----------
  {
    const s = add(14);
    heading(s, chap("①", "競合 ─ 人が入力・人が見て判断・自動で判断を渡す"), "施工管理アプリ・産業用ヘッドセットとは競わずにつなぐ。強みは電気工事の判断の記録と監修");
    const mx = X0, my = 1.15, mw = 4.35, mh = 3.6;
    txt(s, "ポジショニングマップ", { x: mx, y: my, w: mw, h: 0.26, size: 10.5, bold: true });
    const px = mx + 0.3, py = my + 0.42, pw = mw - 0.4, ph = mh - 0.9;
    box(s, { x: px + pw / 2, y: py, w: pw / 2, h: ph / 2, fill: K.fill, name: "target-quadrant" });
    hline(s, px, py + ph / 2, pw, K.gray500, T.stroke.rule);
    s.addShape("line", { x: px + pw / 2, y: py, w: 0, h: ph, line: { color: K.gray500, width: T.stroke.rule } });
    txt(s, "判断を渡す・教える", { x: px + pw / 2 - 1.0, y: py - 0.24, w: 2.0, h: 0.2, size: 8.5, color: K.gray700, align: "center" });
    txt(s, "記録して終わり", { x: px + pw / 2 - 1.0, y: py + ph + 0.04, w: 2.0, h: 0.2, size: 8.5, color: K.gray700, align: "center" });
    txt(s, "← 人が撮って入力", { x: px, y: py + ph / 2 + 0.03, w: 1.4, h: 0.2, size: 8, color: K.gray700 });
    txt(s, "自動で記録 →", { x: px + pw - 1.4, y: py + ph / 2 + 0.03, w: 1.4, h: 0.2, size: 8, color: K.gray700, align: "right" });
    const dots = [
      ["ANDPAD（施工管理）", 0.12, 0.66, false, "r"],
      ["SPIDERPLUS（図面・検査）", 0.3, 0.56, false, "r"],
      ["蔵衛門（工事写真）", 0.08, 0.86, false, "r"],
      ["tebiki（手順の器）", 0.14, 0.32, false, "r"],
      ["日立PS（受変電の点検AI）", 0.44, 0.16, false, "l"],
      ["遠隔臨場カメラ", 0.8, 0.88, false, "l"],
      ["RealWear（遠隔支援）", 0.7, 0.42, false, "l"],
      ["日立 AI Safety（保守）", 0.82, 0.3, false, "l"],
      ["カンツギ", 0.9, 0.18, true, "l"],
      ["社内の熟練者（属人）", 0.58, 0.06, false, "l"],
    ];
    dots.forEach(([label, fx, fy, ours, side]) => {
      const cx = px + fx * pw, cy = py + fy * ph, r = ours ? 0.11 : 0.075;
      s.addShape("ellipse", { x: cx - r, y: cy - r, w: 2 * r, h: 2 * r, fill: { color: ours ? K.ink : K.paper }, line: { color: K.ink, width: 1.25 }, objectName: "pos-" + label });
      const o = { size: ours ? 10 : 8, bold: ours, valign: "middle", h: 0.22, w: 1.7 };
      if (side === "r") txt(s, label, { ...o, x: cx + r + 0.04, y: cy - 0.11, align: "left" });
      else txt(s, label, { ...o, x: cx - r - 1.74, y: cy - 0.11, align: "right" });
    });
    const sx = 4.95, sw = 4.65;
    txt(s, "記録・報告・判断のどこまでを扱うか", { x: sx, y: my, w: sw, h: 0.26, size: 10.5, bold: true });
    table(s, [["会社・サービス", "記録", "報告書", "判断・教材"],
      ["ANDPAD・蔵衛門", "人が撮影・入力", "○ 写真台帳", "×"],
      ["SPIDERPLUS", "人が図面に記録", "○ 検査帳票", "×"],
      ["遠隔臨場カメラ（Safie等）", "○ 映像", "×", "× 人が見る"],
      ["RealWear（ヘッドセット）", "○ 両手が空く", "△ 写真・メモ", "× 遠隔の人"],
      ["日立 AI Safety（保守）", "○ ウェアラブル", "△ 開発中", "○ 安全・手順"],
      ["日立PS（受変電の点検）", "人が撮る写真", "△ 計画", "○ ミスを照合"],
      ["tebiki／Teachme Biz", "×", "×", "△ 器のみ"],
      [{ text: "カンツギ", bold: true }, { text: "○ 一人称で自動", bold: true }, { text: "○ 各社の書式で", bold: true }, { text: "○ 照合＋教材", bold: true }],
    ], { x: sx, y: my + 0.3, w: sw, colW: [1.75, 1.0, 0.95, 0.95], rowH: 0.26, size: 7.5, align: "left", strongRows: [8], name: "competitors" });
    const ty = my + 0.3 + 9 * 0.27 + 0.08;
    box(s, { x: sx, y: ty, w: sw, h: 0.78, fill: K.ink, name: "threat" });
    txt(s, [
      run("最大の競合：", { bold: true }), run("日立グループ（AI Safety・日立PS。次のページ）。施工管理アプリ・RealWearも自動報告を加えうる", { breakLine: true }),
      run("方針：", { bold: true }), run("競わずにつなぐ（報告書は各社の形式、端末はRealWearも選べる）。領域は電気工事の施工にずらす"),
    ], { x: sx + 0.12, y: ty, w: sw - 0.24, h: 0.78, size: 7.5, color: K.paper, valign: "middle", psa: 2 });
    txt(s, "出典：各社公表資料・販売店資料、日立ニュースリリース（2025年10月）・Physical AI 事例（2026年7月）。日立PS＝日立パワーソリューションズ", { x: X0, y: 5.0, w: W, h: 0.24, size: S.note, color: K.gray500 });
  }

  // ---------- Slide ①日立との違い ----------
  {
    const s = add(15);
    heading(s, chap("①", "差別化 ─ 日立グループと方向は同じ。だから領域をずらす"), "日立は自社の保守契約にAIを組み込んで稼ぐ。当社は中堅の電気工事会社の施工（書類・技能継承）を月額で");
    const top = 1.12, lw = 4.45, ih = lw * 552 / 1332;
    box(s, { x: X0, y: top, w: lw, h: ih + 0.06, line: K.rule });
    s.addImage({ path: path.join(ASSETS, "ref-hitachi-ai-safety.png"), x: X0 + 0.03, y: top + 0.03, w: lw - 0.06, h: ih });
    txt(s, "引用：日立製作所・日立ビルシステム・日立パワーソリューションズ ニュースリリース（2025年10月10日）「AI Safetyソリューションのイメージ図」", { x: X0, y: top + ih + 0.1, w: lw, h: 0.32, size: 7.5, color: K.gray500 });
    txt(s, [run("日立の強み：", { bold: true, color: ACC, breakLine: true }), run("NVIDIAと組んだ映像AI（VLM）、日立ビルシステムの全国の現場で数千人が運用（2025年11月〜）、ブランドと保守網。技術と規模では勝てない", { fontSize: 8.5 })],
      { x: X0, y: top + ih + 0.45, w: lw, h: 0.85, size: 9, psa: 2 });
    const rx = 5.05, rw = 4.55;
    table(s, [["", "日立 AI Safety", "日立PS 点検AI", "カンツギ"],
      ["仕事", "ビル設備の保守（昇降機など）", "受変電設備（VCB）の点検", { text: "電気工事の施工", bold: true }],
      ["使う人", "日立の自社技術者", "日立PSの保守員", { text: "中堅の電気工事会社", bold: true }],
      ["AIの働き", "映像で危険を通知", "点検前後の写真でミスを照合", { text: "記録→書類→教材", bold: true }],
      ["稼ぎ方", "保守契約・ビル向けサービス", "保守契約（AIは品質と原価のため）", { text: "月額の利用料", bold: true }],
      ["段階", "現場で運用中", "検証中", { text: "準備期", bold: true }],
      ["知識の型", "日立の保守ナレッジ", "熟練者の判断をAIへ（個別開発）", { text: "電気工事の工種の型", bold: true }],
      ["端末", "ウェアラブル＋スマホ", "写真（カメラ）", { text: "普段使いのAIグラス", bold: true }]],
      { x: rx, y: top, w: rw, colW: [0.7, 1.25, 1.3, 1.3], rowH: 0.34, size: 7.5, leftCols: [1, 2, 3], align: "left", name: "vs-hitachi" });
    box(s, { x: X0, y: 4.0, w: W, h: 0.88, fill: K.ink, name: "diff-strategy" });
    txt(s, [run("差別化戦略：", { bold: true, color: L.YELLOW }), run("①安全と保守では競わない（日立の領域）　②施工の書類（写真台帳・検査記録）を発注者の書式で出す　③電気工事の工種の型と教材を業界共通で先に揃える", { breakLine: true }),
      run("もし日立が施工・中堅向けに来たら：", { bold: true, color: L.YELLOW }), run("競わず提携も選ぶ（日立の基盤に電気工事の型を載せる・協力会社へ広げる）。2〜3年で型とデータを先に揃える")],
      { x: X0 + 0.15, y: 4.0, w: W - 0.3, h: 0.88, size: 8.5, color: K.paper, valign: "middle", psa: 2 });
    txt(s, "出典：日立ニュースリリース（2025年3月・10月・11月）、日立 Physical AI 事例（2026年7月）、BUILT・ITmedia。日立の価格・外販の予定は未確認", { x: X0, y: 4.98, w: W, h: 0.26, size: S.note, color: K.gray500 });
  }

  // ---------- Slide 12 ⑦チーム・実行力 ----------
  {
    const s = add(16);
    detailRef(s, DP.team);
    heading(s, chap("②", `スタッフ ─ 5年目${Y5.staff}人、現場監修とコンサル部隊を内製`), "人数・特性・任せる役割、そして自分の役割。帳票の設定は外部に委託せず、社内に知見を貯める");
    const lx = X0, lw = 3.4, top = 1.15;
    box(s, { x: lx, y: top, w: lw, h: 3.7, fill: K.ink, name: "founder" });
    txt(s, [run("代表取締役 CEO", { fontSize: 9, breakLine: true }), run("代表（創業者）", { bold: true, fontSize: 14 })], { x: lx + 0.15, y: top + 0.1, w: lw - 0.3, h: 0.6, color: K.paper });
    txt(s, [
      run("電線メーカー 電力事業部門で、66kV／275kV 地中送電線の施工管理5年", { breakLine: true }),
      run("Python・Power BI・FastAPI・Docker・LangChain・RAG を独学し業務に実装", { breakLine: true }),
      run("ニュース要約→音声合成→毎朝配信の仕組みを個人で構築・運用中", { breakLine: true }),
      run("1級電気工事施工管理技士・G検定・大学院修了", { breakLine: true }),
      run("私の役割：最初の営業・現場監修の統括・理念と判断の基準づくり", { bold: true, color: L.YELLOW }),
    ], { x: lx + 0.15, y: top + 0.78, w: lw - 0.3, h: 2.85, size: 9.5, color: K.paper, psa: 6 });
    const rx = 3.95, rw = 5.65;
    txt(s, "チームと外部の体制", { x: rx, y: top, w: rw, h: 0.26, size: 10.5, bold: true });
    table(s, [
      ["役割", "人数 1→5年目", "任せる仕事", "求める特性"],
      ...V.ROLES.slice(1).map((r) => [r[0].replace(/（.*?）/, ""), `${r[2][0]}→${r[2][4]}人`, r[3].replace(/（当社のコンサル部隊）/, ""), r[4]]),
      ["社外顧問", "2人", "映像AI（大学）・電気工事業界のOB", "月1〜2回の助言"],
    ], { x: rx, y: top + 0.3, w: rw, colW: [1.15, 0.7, 2.2, 1.6], rowH: 0.27, size: 7.5, leftCols: [2, 3], name: "team" });
    box(s, { x: rx, y: top + 3.2, w: rw, h: 0.5, fill: K.fill, name: "proof" });
    txt(s, [run("コンサル部隊を社内に持つ理由：", { bold: true }), run("各社の帳票や連携の設定を外部に任せると、知見が社外に残る。社内で行い、型として製品に戻す")], { x: rx + 0.15, y: top + 3.2, w: rw - 0.3, h: 0.5, size: 9, valign: "middle" });
    txt(s, `パートナー：グラスのメーカー・リース会社・大学。役割別の年収と採用の時期は詳細版 p.17。実行力の証拠：音声の自動配信は毎朝稼働中、デモアプリも自作`, { x: X0, y: 4.98, w: W, h: 0.26, size: S.note, color: K.gray500 });
  }

  // ---------- Slide ③経営理念 ----------
  {
    const s = add(17);
    heading(s, chap("③", "経営理念 ─ 現場の勘を次の担い手へ"), "ホームページで多くの人に伝える内容：使命 → 大切にすること → 一人ひとりへの約束");
    const top = 1.15, lw = 3.6;
    box(s, { x: X0, y: top, w: lw, h: 3.72, fill: K.ink, name: "mission" });
    txt(s, [run("使命（ミッション）", { bold: true, color: L.YELLOW, fontSize: 10, breakLine: true }),
      run("現場の勘を", { bold: true, fontSize: 20, breakLine: true }), run("次の担い手へ", { bold: true, fontSize: 20, breakLine: true }),
      run("熟練者の判断をデータで継ぎ、人が減っても日本の現場の品質を守る", { fontSize: 10.5, breakLine: true }),
      run(" ", { fontSize: 6, breakLine: true }),
      run("目指す姿（ビジョン）", { bold: true, color: L.YELLOW, fontSize: 10, breakLine: true }),
      run("人が減っても、現場が迷わず回る。若手が「聞ける先輩がいない」と困らない", { fontSize: 10.5 })],
      { x: X0 + 0.2, y: top + 0.1, w: lw - 0.4, h: 3.52, color: K.paper, valign: "middle", psa: 3 });
    const rx = X0 + lw + 0.25, rw = W - lw - 0.25;
    txt(s, "大切にする3つのこと（バリュー）", { x: rx, y: top, w: rw, h: 0.26, size: 10.5, bold: true });
    [["現場を主役に", "作業の手を止めない。入力を増やさない。現場の言葉で話す"], ["勘に敬意を", "記録は本人の同意のもとで。誰の技術かを残し、評価につなげる"], ["数字で約束する", "効果は毎月測って見せる。効かなければ正直に言う"]].forEach(([h, d], i) => {
      const y = top + 0.32 + i * 0.5;
      box(s, { x: rx, y: y + 0.04, w: 0.36, h: 0.38, fill: ACC });
      txt(s, String(i + 1), { x: rx, y: y + 0.04, w: 0.36, h: 0.38, size: 12, bold: true, color: K.paper, align: "center", valign: "middle" });
      txt(s, [run(h, { bold: true, breakLine: true }), run(d, { fontSize: 8.5, color: K.gray700 })], { x: rx + 0.45, y, w: rw - 0.45, h: 0.46, size: 10, valign: "middle" });
    });
    txt(s, "一人ひとりへの約束（ホームページの3つの入口）", { x: rx, y: top + 1.95, w: rw, h: 0.26, size: 10.5, bold: true });
    table(s, [["誰に", "伝えること"],
      ["お客様（工事会社）", "書類の時間を減らし、効果を毎月の数字で示す"],
      ["現場で働く人", "監視ではない。あなたの技術を残し、若手と評価につなげる"],
      ["社会・次の世代", "現場の勘を業界の財産にし、AIとロボットに教える"]],
      { x: rx, y: top + 2.25, w: rw, colW: [1.55, rw - 1.55], rowH: 0.36, size: 9, leftCols: [1], name: "promise" });
    txt(s, "トップページの一文：「ベテランの『ここが大事』を、メガネ型のカメラとAIで残し、報告書と若手の教材に変えます」", { x: X0, y: 4.98, w: W, h: 0.26, size: S.note, color: K.gray500 });
  }

  // ---------- Slide 7 ④市場規模 ----------
  {
    const s = add(18);
    detailRef(s, DP.market);
    heading(s, chap("④", "市場規模 ─ ターゲット市場は約141億円"), "全体市場（年7〜9%成長）→ ターゲット市場（顧客数×単価×年1回）→ 自社が狙う市場");
    const cx = 2.3, cy = 3.05;
    s.addShape("ellipse", { x: cx - 1.9, y: cy - 1.85, w: 3.8, h: 3.7, fill: { color: K.fill }, line: { color: K.gray500, width: 1 }, objectName: "tam" });
    s.addShape("ellipse", { x: cx - 1.2, y: cy - 0.75, w: 2.4, h: 2.5, fill: { color: K.gray700 }, line: { type: "none" }, objectName: "sam" });
    s.addShape("ellipse", { x: cx - 0.6, y: cy + 0.35, w: 1.2, h: 1.2, fill: { color: K.ink }, line: { type: "none" }, objectName: "som" });
    txt(s, [run("全体市場", { bold: true, breakLine: true }), run("建設テック＋スマートファクトリー", { fontSize: 8.5 })], { x: cx - 1.6, y: cy - 1.7, w: 3.2, h: 0.6, size: 11, align: "center" });
    txt(s, [run("ターゲット市場", { bold: true, breakLine: true }), run("約141億円", { bold: true, breakLine: true }), run("約3,100社", { fontSize: 8.5 })], { x: cx - 1.1, y: cy - 0.62, w: 2.2, h: 0.8, size: 11, color: K.paper, align: "center" });
    txt(s, [run("自社", { bold: true, breakLine: true }), run(`${(Y5.sites * 300 / 1e4).toFixed(2)}億円`, { fontSize: 9 })], { x: cx - 0.6, y: cy + 0.55, w: 1.2, h: 0.8, size: 10.5, color: K.paper, align: "center", valign: "middle" });
    const rx = 4.7, rw = 4.9;
    txt(s, "ターゲット市場の数え方（顧客数×年額、億円）", { x: rx, y: 1.15, w: rw, h: 0.26, size: 10.5, bold: true });
    table(s, [
      ["業種＼規模", "大（300人以上）", "中（30〜299人）", "小（30人未満）"],
      ["電気工事", "300社×1,500万＝45", { text: "2,100社×300万＝63", bold: true, fill: L.PALEY }, { text: "対象外", color: K.gray500 }],
      ["電気通信工事", "100社×1,500万＝15", "600社×300万＝18", { text: "対象外", color: K.gray500 }],
      [{ text: "電線・素材工場", color: K.gray700 }, { text: "隣接市場（成長期以降）", color: K.gray700 }, { text: "", color: K.gray700 }, { text: "", color: K.gray700 }],
    ], { x: rx, y: 1.45, w: rw, colW: [1.1, 1.3, 1.4, 1.1], rowH: 0.34, size: 8, align: "center", name: "segments" });
    txt(s, "色付き＝最初の顧客。大手は支店5拠点分。小規模は1年で元が取れる条件（月105人時）を満たしにくい", { x: rx, y: 2.86, w: rw, h: 0.36, size: 8, color: K.gray700 });
    box(s, { x: rx, y: 3.3, w: rw, h: 1.38, fill: K.ink, name: "growth" });
    txt(s, [
      run("全体市場の伸び", { bold: true, fontSize: 10.5, breakLine: true }),
      run("国内建設テック（建築分野）：1,845億円→3,043億円、年7.4%（2023→2030年度）(1)", { breakLine: true }),
      run("国内スマートファクトリー：42億→92億米ドル、年9.03%（2025→2034年）(2)", { breakLine: true }),
      run(`自社が狙う市場：5年目${Y5.sites}拠点×年300万円＝約${(Y5.sites * 300 / 1e4).toFixed(2)}億円（ターゲット市場の約${(Y5.sites * 300 / 141e2).toFixed(1)}%）`),
    ], { x: rx + 0.15, y: 3.35, w: rw - 0.3, h: 1.28, size: 8.5, color: K.paper, psa: 2, valign: "middle" });
    txt(s, "(1) 矢野経済研究所 プレスリリースNo.3789（2025年4月）　(2) IMARC Group「Japan Smart Factory Market」　社数：電気工事業の許可業者65,497社（令和7年3月末、国土交通省）から規模別に推計（内訳は仮定）", { x: X0, y: 4.9, w: W, h: 0.3, size: S.note, color: K.gray500 });
  }

  // ---------- Slide 8 ④ターゲット ----------
  {
    const s = add(19);
    detailRef(s, DP.target);
    heading(s, chap("④", "ターゲット ─ 最初の顧客は関東の中堅電気工事会社"), "市場の大きさ・困りごとの強さ・会いに行きやすさ・波及効果の4つで選んだ");
    const top = 1.15;
    txt(s, "セグメントの評価（5点満点）", { x: X0, y: top, w: 5.3, h: 0.26, size: 10.5, bold: true });
    const ev = [
      ["電気工事・中堅", 5, 5, 4, 3],
      ["電気工事・大手", 4, 4, 2, 5],
      ["電線工場・中堅", 1, 4, 4, 3],
      ["電気通信工事・中堅", 3, 4, 2, 3],
    ];
    table(s, [["候補", "大きさ", "困りごと", "会いやすさ", "波及", "合計"],
      ...ev.map((r, i) => r.map((v, j) => (j === 0 ? { text: v, bold: i === 0 } : { text: String(v), bold: i === 0 })).concat([{ text: String(r.slice(1).reduce((a, b) => a + b, 0)), bold: true, color: i === 0 ? ACC : K.ink }]))],
      { x: X0, y: top + 0.3, w: 5.3, colW: [1.7, 0.7, 0.75, 0.8, 0.6, 0.75], rowH: 0.32, size: 9, align: "center", strongRows: [1], name: "seg-eval" });
    txt(s, [
      run("困りごと：", { bold: true }), run("2024年から残業に上限。工事写真と報告書が多い", { breakLine: true }),
      run("会いやすさ：", { bold: true }), run("施工管理5年の経歴が信用になる。川崎から関東へ日帰り", { breakLine: true }),
      run("波及：", { bold: true }), run("大手が採用すると協力会社が同じ形式に合わせる → 大手1社を目玉の事例に"),
    ], { x: X0, y: top + 2.0, w: 5.3, h: 0.95, size: 9, psa: 3 });
    const rx = 6.0, rw = 3.6;
    box(s, { x: rx, y: top, w: rw, h: 1.5, fill: K.ink, name: "icp" });
    txt(s, [run("理想の顧客像", { bold: true, color: L.YELLOW, breakLine: true }),
      run("関東の電気工事会社", { bold: true, breakLine: true }),
      run("施工管理者30人以上・電力や公共工事が多く書類が多い・残業規制で人を増やせない")], { x: rx + 0.15, y: top + 0.05, w: rw - 0.3, h: 1.4, size: 9.5, color: K.paper, valign: "middle", psa: 3 });
    txt(s, "意思決定に関わる3人", { x: rx, y: top + 1.62, w: rw, h: 0.26, size: 10.5, bold: true });
    [["決裁", "社長・工事部長", "回収期間と残業の削減"], ["使う", "施工管理者・電工", "入力が増えないか"], ["推進", "DX担当者", "定着と効果の説明"]].forEach(([r, who, care], i) => {
      const y = top + 1.92 + i * 0.4;
      box(s, { x: rx, y: y + 0.04, w: 0.55, h: 0.3, fill: i === 0 ? ACC : K.gray700 });
      txt(s, r, { x: rx, y: y + 0.04, w: 0.55, h: 0.3, size: 9.5, bold: true, color: K.paper, align: "center", valign: "middle" });
      txt(s, [run(who, { bold: true, breakLine: true }), run("気にすること：" + care, { fontSize: 8, color: K.gray700 })], { x: rx + 0.62, y, w: rw - 0.62, h: 0.38, size: 9, valign: "middle" });
    });
    txt(s, "評価は社内の仮採点。最初の3社への導入で「困りごと」と「会いやすさ」を検証する", { x: X0, y: 4.98, w: W, h: 0.26, size: S.note, color: K.gray500 });
  }

  // ---------- Slide 9 ⑤ビジネスモデルの3要素 ----------
  {
    const s = add(20);
    detailRef(s, DP.price);
    heading(s, "④ ビジネスモデル ─ 誰に・どう実現し・どう稼ぐか", "戦略モデル（顧客価値）・オペレーションモデル（実現の方法）・マネタイズモデル（収益の方法）");
    const top = 1.12, lw = 1.6;
    const row = (y, h, title, sub) => {
      box(s, { x: X0, y, w: lw, h, fill: K.ink });
      txt(s, [run(title, { bold: true, fontSize: 12, breakLine: true }), run(sub, { fontSize: 8.5 })], { x: X0 + 0.1, y, w: lw - 0.2, h, color: K.paper, valign: "middle" });
    };
    row(top, 0.75, "戦略モデル", "誰に・どんな価値を");
    txt(s, [run("関東の中堅電気工事会社に、", { breakLine: true }), run("「書類の時間を減らし、熟練者の判断を若手に残す」", { bold: true, color: ACC })], { x: X0 + lw + 0.15, y: top, w: W - lw - 0.15, h: 0.75, size: 11, valign: "middle" });
    const oy = top + 0.85, oh = 2.05;
    row(oy, oh, "オペレーション", "価値を実現する流れ");
    const steps = [["獲得", "デモ・紹介\n展示会"], ["導入", "2か月\n工種の型を設定"], ["記録", "AIグラスで\n要所を自動記録"], ["整理・監修", "AIが整理\n施工経験者が確認"], ["納品", "報告書・照合\n・教材"], ["定着", "月次で\n効果を測定"]];
    const sx = X0 + lw + 0.15, sw = (W - lw - 0.15) / 6;
    steps.forEach(([t, d], i) => {
      const x = sx + i * sw;
      s.addShape("chevron", { x, y: oy, w: sw - 0.02, h: 0.42, fill: { color: i === 3 ? ACC : K.gray700 }, line: { type: "none" } });
      txt(s, t, { x: x + 0.1, y: oy, w: sw - 0.2, h: 0.42, size: 9.5, bold: true, color: K.paper, align: "center", valign: "middle" });
      txt(s, d, { x, y: oy + 0.48, w: sw - 0.05, h: 0.5, size: 8.5, align: "center" });
    });
    table(s, [["担う人・資源", "能力・約束"], ["導入コンサルタント（当社のコンサル部隊）", "1人で年6拠点を立ち上げ、各社の帳票と連携を設定。知見を製品に戻す"], ["施工経験者による監修", "判断の意味づけ。真似されにくさの源泉"], ["パートナー", "グラスのメーカー・リース会社・クラウド・大学（映像AI）"]],
      { x: sx, y: oy + 1.02, w: W - lw - 0.15, colW: [2.4, 5.05], rowH: 0.25, size: 8, leftCols: [1], name: "ops" });
    const my = oy + oh + 0.1;
    row(my, 0.85, "マネタイズ", "どう稼ぐ・いつ");
    txt(s, [run("基盤 月5万円＋端末 月2万円×台数（標準10台で年300万円・年額前払い）", { bold: true }), run("＋初年度だけ導入支援200万円（着手金50%）", { breakLine: true }), run(`お客様は1台1日${be}分減れば元が取れる。当社の黒字化は1拠点15台が条件（p.26）`)], { x: X0 + lw + 0.15, y: my, w: W - lw - 0.15, h: 0.85, size: 10, valign: "middle", psa: 3 });
  }

  // ---------- Slide 4 収益の仕組み ----------
  {
    const s = add(21);
    detailRef(s, DP.price + "・" + DP.ltv + "・" + DP.data);
    heading(s, "④ 収益モデル ─ 「会社の基盤」＋「端末の台数」で払う", "毎年積み上がる利用料と、初年度だけの導入支援の二本。報告書の形に整えるところまで利用料に含む");
    const lx = X0, lw = 4.45, top = 1.15;
    txt(s, "1拠点（10台）が当社に払う金額（万円）", { x: lx, y: top, w: lw, h: 0.26, size: 10.5, bold: true });
    const k = (lw - 1.3) / 500;
    const segs = [["基盤", 60, K.ink, K.paper], ["端末×10台", 240, K.gray700, K.paper], ["導入支援", 200, K.paper, K.ink]];
    [["初年度", 500, segs], ["2年目以降", 300, segs.slice(0, 2)]].forEach(([lab, total, ss], r) => {
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
      ["基盤", "月5万円／拠点", "会社ごとのデータ保管庫・管理画面・図面と手順・各社の報告書の型・月次の効果レポート"],
      ["端末ライセンス", "月2万円／台", "記録→写真の自動抽出→日報・写真台帳を各社の書式で。AIで解析する場面に月の上限"],
      ["端末本体", "当社の売上外", `お客様がリース（AIグラス8台＋表示付き2台で年約${man(V.leaseSite())}万円）`],
      ["導入支援", "200万円（初年度）", "導入コンサルタントが約2か月・42人日、現場に入る（内訳は詳細版 p.13）"],
    ].forEach(([n, p, d], i) => {
      const y = dy + i * 0.36;
      hline(s, lx, y, lw, K.rule, T.stroke.rule);
      txt(s, [run(n, { bold: true, breakLine: true }), run(p, { fontSize: 8, color: K.gray700 })], { x: lx, y: y + 0.02, w: 1.45, h: 0.34, size: 9, valign: "middle" });
      txt(s, d, { x: lx + 1.5, y, w: lw - 1.5, h: 0.36, size: 8.5, valign: "middle" });
    });
    hline(s, lx, dy + 4 * 0.36, lw, K.rule, T.stroke.rule);
    box(s, { x: lx, y: dy + 1.52, w: lw, h: 0.78, fill: K.ink });
    txt(s, [run(`1拠点の粗利 年${man(U.gpYear)}万円`, { bold: true }), run(`（300万円−AI・保存${man(V.cogsSite())}万円＝粗利率${Math.round(V.marginSite() * 100)}%）`, { breakLine: true }),
      run(`LTV 約${man(U.ltv(7))}万円`, { bold: true }), run(`（${man(U.gpYear)}万円×7年＋導入支援の粗利${man(U.setupGp)}万円）`, { breakLine: true }),
      run(`CAC 約${man(U.cac)}万円`, { bold: true }), run(`（営業${SALES}人${man(SALES * U.salesPay)}万円＋広告500万円）÷新規33拠点　LTV/CAC 約${Math.round(U.ltv(7) / U.cac)}倍`)],
      { x: lx + 0.12, y: dy + 1.52, w: lw - 0.24, h: 0.78, size: 8, color: K.paper, valign: "middle", psa: 1 });

    const rx = 5.05, rw = 4.55;
    txt(s, "お客様から見た回収（1拠点10台・累計・万円）", { x: rx, y: top, w: rw, h: 0.26, size: 10.5, bold: true });
    const months = Array.from({ length: 25 }, (_, m) => String(m));
    const gain = Array.from({ length: 25 }, (_, m) => Math.round(m * CV.docs / 12));
    const firstYear = CV.cost + V.PLAN.setup;
    const cost = Array.from({ length: 25 }, (_, m) => Math.round(m < 12 ? firstYear : firstYear + CV.cost));
    const axis = { catAxisLabelColor: K.gray700, valAxisLabelColor: K.gray700, catAxisLabelFontSize: 8, valAxisLabelFontSize: 8,
      catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", valGridLine: { color: K.rule, size: 0.5 }, catGridLine: { style: "none" },
      valAxisMinVal: 0, valAxisMaxVal: 1000, valAxisMajorUnit: 200, catAxisLineShow: true, valAxisLineShow: false, showLegend: false };
    s.addChart([
      { type: pres.charts.LINE, data: [{ name: "効果の累計", labels: months, values: gain }], options: { chartColors: [ACC], lineSize: 3, lineDataSymbol: "none" } },
      { type: pres.charts.LINE, data: [{ name: "払った額の累計", labels: months, values: cost }], options: { chartColors: [K.ink], lineSize: 2, lineDash: "dash", lineDataSymbol: "none" } },
    ], { x: rx, y: top + 0.28, w: rw, h: 2.5, ...axis, objectName: "roi-chart" });
    txt(s, [run("━ 効果の累計", { bold: true, color: ACC }), run("　┅ 支払いの累計（横軸は月）", { color: K.gray700 })], { x: rx + 0.5, y: top + 0.32, w: 3.5, h: 0.22, size: 8.5 });
    txt(s, `初年度 約${man(firstYear)}万円（利用料300＋導入支援200＋リース${man(V.leaseSite())}）÷ 月約${man(CV.docs / 12)}万円 ＝ 約${Math.round(firstYear / (CV.docs / 12))}か月で回収。2年目からは毎年 得${man(CV.docs)}万円／払う${man(CV.cost)}万円`, { x: rx, y: top + 2.82, w: rw, h: 0.5, size: 8.5, bold: true });
    txt(s, `月約${man(CV.docs / 12)}万円＝10台×20日×（職長15分＋施工管理者20分）。手戻りも実測できれば約${Math.round(firstYear / ((CV.docs + CV.rework) / 12))}か月で回収`, { x: rx, y: top + 3.35, w: rw, h: 0.4, size: S.note, color: K.gray700 });
  }

  // ---------- Slide 11 ⑤売り方 ----------
  {
    const s = add(22);
    detailRef(s, DP.target + "・" + DP.sales);
    heading(s, "④ 売り方 ─ 最初の10社は、業界団体・展示会・提携で連れてくる", "デモ → 3か月の有償試験導入 → 本契約。大手1社を目玉の事例にして協力会社へ広げる");
    const top = 1.15, lw = 4.2;
    txt(s, "5年間の積み上げ（顧客側から）", { x: X0, y: top, w: lw, h: 0.26, size: 10.5, bold: true });
    const fun = [["関東の中堅電気工事会社", "735社", 1.0], ["4つの条件に合う（約25%）", "180社", 0.82], ["会える（60%）", "108社", 0.62], ["本契約（30%）", "32社", 0.42]];
    fun.forEach(([a, b, k], i) => {
      const y = top + 0.35 + i * 0.5, w = lw * k, x = X0 + (lw - w) / 2;
      box(s, { x, y, w, h: 0.42, fill: i === 3 ? ACC : i === 0 ? K.gray500 : K.gray700 });
      txt(s, [run(a + "  "), run(b, { bold: true })], { x, y, w, h: 0.42, size: 9.5, color: K.paper, align: "center", valign: "middle" });
    });
    txt(s, [run("＋ それ以外555社から29社（会える35%・本契約15%）", { bold: true, breakLine: true }), run(`×1社1.3拠点＋大手1社5拠点＝約85拠点。解約を引いた計画は${Y5.sites}拠点`)], { x: X0, y: top + 2.4, w: lw, h: 0.5, size: 9.5, align: "center" });
    const rx = 4.85, rw = 4.75;
    txt(s, "接点と役割", { x: rx, y: top, w: rw, h: 0.26, size: 10.5, bold: true });
    table(s, [["接点", "何をするか"],
      ["業界団体・地域の工業組合", "勉強会で「2024年問題と書類の削減」を話す"],
      ["電設の展示会", "デモアプリで、自分の工事で試す体験"],
      ["グラスのメーカー・販売店", "グラスとセットで提案（代理店）"],
      ["施工管理アプリとの連携", "各社の形式で出力。競わず並べて使う"],
      ["大手電気工事会社1社", "目玉の事例。協力会社へ同じ形式で広がる"],
    ], { x: rx, y: top + 0.3, w: rw, colW: [1.85, 2.9], rowH: 0.3, size: 8.5, leftCols: [1], name: "channels" });
    box(s, { x: X0, y: 3.55, w: W, h: 0.65, fill: K.fill });
    const st = [["デモ", "1か月"], ["有償試験導入", "3か月・導入支援200万円"], ["本契約", "年300万円・年額前払い"], ["紹介", "協力会社・同業へ"]];
    st.forEach(([a, b], i) => {
      const x = X0 + 0.15 + i * 2.27;
      txt(s, [run(a, { bold: true, color: ACC, breakLine: true }), run(b, { fontSize: 8.5 })], { x, y: 3.58, w: 2.0, h: 0.6, size: 10.5, valign: "middle" });
      if (i < 3) arrow(s, x + 1.95, 3.88, 0.25, 0, { lw: 2 });
    });
    box(s, { x: X0, y: 4.3, w: W, h: 0.55, fill: K.ink });
    txt(s, [run("最初の10社：", { bold: true, color: L.YELLOW }), run("準備期に有償試験3社 → 1年目に本契約3社＋試験導入 → 2年目に累計12拠点。商談期間は4〜6か月を見込む")], { x: X0 + 0.15, y: 4.3, w: W - 0.3, h: 0.55, size: 9.5, color: K.paper, valign: "middle" });
    txt(s, "4つの条件：施工管理者30人以上・電力や官公庁の工事が多い・手戻りが高くつく工種がある・デジタル化の素地。率はすべて仮定で、最初の3社で実測する", { x: X0, y: 4.98, w: W, h: 0.26, size: S.note, color: K.gray500 });
  }

  // ---------- Slide 9 ⑥売上予測・費用 ----------
  {
    const s = add(23);
    detailRef(s, DP.sales + "・" + DP.pl);
    heading(s, chap("④", `売上予測 ─ 拠点×月25万円。5年目は月約${man(Math.round(Y5.sales / 12 / 10) * 10)}万円`), "利用料が毎年積み上がり（解約は年10%で引く）、導入支援と個社設定は新規拠点の数に比例する");
    const years = ["1年目", "2年目", "3年目", "4年目", "5年目"];
    const gx = X0, gw = 4.45, top = 1.15;
    txt(s, "成長曲線（百万円）", { x: gx, y: top, w: gw, h: 0.26, size: 10.5, bold: true });
    txt(s, [run("━ 売上高", { bold: true }), run("　■ 営業利益", { color: K.gray700 }), run("　┅ 累積営業損益", { color: K.gray700 })], { x: gx + 1.4, y: top, w: gw - 1.4, h: 0.26, size: 8, align: "right", valign: "middle" });
    s.addChart([
      { type: pres.charts.LINE, data: [{ name: "売上高", labels: years, values: PR.map((r) => mil(r.sales)) }], options: { chartColors: [K.ink], lineSize: 3.5, lineDataSymbol: "circle", lineDataSymbolSize: 6, showValue: true } },
      { type: pres.charts.LINE, data: [{ name: "営業利益", labels: years, values: PR.map((r) => mil(r.op)) }], options: { chartColors: [K.gray700], lineSize: 2, lineDataSymbol: "square", lineDataSymbolSize: 5, showValue: true } },
      { type: pres.charts.LINE, data: [{ name: "累積営業損益", labels: years, values: PR.map((r) => mil(r.cum)) }], options: { chartColors: [K.gray500], lineSize: 2, lineDash: "dash", lineDataSymbol: "none", showValue: false } },
    ], { x: gx, y: top + 0.28, w: gw, h: 2.3, catAxisLabelColor: K.gray700, valAxisLabelColor: K.gray700, catAxisLabelFontSize: 9, valAxisLabelFontSize: 8,
      catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", valGridLine: { color: K.rule, size: 0.5 }, catGridLine: { style: "none" },
      valAxisMinVal: -200, valAxisMaxVal: 300, valAxisMajorUnit: 50, valAxisLineShow: false, showLegend: false,
      dataLabelFontSize: 8, dataLabelFontFace: "+mn-lt", dataLabelColor: K.ink, dataLabelPosition: "t", catAxisLabelPos: "low", objectName: "growth-chart" });
    box(s, { x: gx, y: 3.85, w: gw, h: 1.05, fill: K.ink, name: "monthly" });
    txt(s, [
      run(`月次売上の根拠（5年目の月平均 約${man(Y5.sales / 12)}万円）`, { bold: true, fontSize: 10, breakLine: true }),
      run(`利用料 約${man(Y5.license / 12)}万円 ＝ 月25万円×（継続${Y5.prev - Y5.ch}拠点＋新規${Y5.nw}拠点×半年）`, { breakLine: true }),
      run(`導入支援 約${man(Y5.setup / 12)}万円 ＝ 200万円×新規${Y5.nw}拠点÷12`, { breakLine: true }),
      run(`個社設定（大手の基幹連携など） 約${man(Y5.contract / 12)}万円`),
    ], { x: gx + 0.15, y: 3.88, w: gw - 0.3, h: 1.0, size: 8.5, color: K.paper, psa: 1 });

    const rx = 5.05, rw = 4.55;
    txt(s, "費用の内訳（万円）", { x: rx, y: top, w: rw, h: 0.26, size: 10.5, bold: true });
    table(s, [
      ["", "1年目", "3年目", "5年目"],
      ["社員数", ...[0, 2, 4].map((i) => PR[i].staff + "名")],
      ["人件費（販管費分・会社負担込み）", ...[0, 2, 4].map((i) => man(PR[i].payroll - PR[i].transfer))],
      ["採用費", ...[0, 2, 4].map((i) => man(PR[i].recruit))],
      ["オフィス（賃料＋共益費）", ...[0, 2, 4].map((i) => man(V.GROWTH.other.office[i]))],
      ["広告・展示会", ...[0, 2, 4].map((i) => man(V.GROWTH.other.ads[i]))],
      ["交通（営業・定着支援）", ...[0, 2, 4].map((i) => man(V.GROWTH.other.travel[i]))],
      ["端末・ツール", ...[0, 2, 4].map((i) => man(V.GROWTH.other.tools[i]))],
      ["専門家・社外顧問", ...[0, 2, 4].map((i) => man(V.GROWTH.other.experts[i] + V.ADVISORS))],
      ["AIの学習（計算費）", ...[0, 2, 4].map((i) => man(V.GROWTH.other.ml[i]))],
      ["雑費", ...[0, 2, 4].map((i) => man(V.GROWTH.other.misc[i]))],
      ["販管費 合計", ...[0, 2, 4].map((i) => man(PR[i].sga))],
    ], { x: rx, y: top + 0.3, w: rw, colW: [2.05, 0.83, 0.83, 0.84], rowH: 0.25, size: 8, strongRows: [11], name: "costs" });
    txt(s, "所在地案：川崎市（関東一円へ日帰り）。人件費は役割別の年収×1.2。導入支援・個社設定に回った人件費は原価へ", { x: rx, y: 4.6, w: rw, h: 0.4, size: S.note, color: K.gray500 });
    txt(s, "売上＝既存の利用料（（前年末拠点−解約）×300万円）＋新規の利用料（新規×300万円×0.5）＋導入支援（新規×200万円）＋個社設定", { x: X0, y: 5.04, w: W, h: 0.22, size: S.note, color: K.gray500 });
  }

  // ---------- Slide 10 ⑥損益分岐点 ----------
  {
    const s = add(24);
    detailRef(s, DP.bep);
    heading(s, chap("④", `損益分岐点 ─ 5年目で分岐点の${Math.round(Y5.sales / Y5.bep * 100)}%。あと約${gapSites}拠点分`), "損益分岐点売上＝販管費（固定費）÷粗利率。10台のままでは5年目でも届かない");
    const years = ["1年目", "2年目", "3年目", "4年目", "5年目"];
    const gx = X0, gw = 5.2, top = 1.15;
    txt(s, [run("━ 実際の売上", { bold: true }), run("　┅ 損益分岐点売上（百万円）", { color: K.gray700 })], { x: gx, y: top, w: gw, h: 0.26, size: 8.5, valign: "middle" });
    s.addChart([
      { type: pres.charts.LINE, data: [{ name: "売上高", labels: years, values: PR.map((r) => mil(r.sales)) }], options: { chartColors: [K.ink], lineSize: 3, lineDataSymbol: "circle", lineDataSymbolSize: 6, showValue: true, dataLabelPosition: "b" } },
      { type: pres.charts.LINE, data: [{ name: "損益分岐点売上", labels: years, values: PR.map((r) => mil(r.bep)) }], options: { chartColors: [K.gray500], lineSize: 2, lineDash: "dash", lineDataSymbol: "diamond", lineDataSymbolSize: 6, showValue: true, dataLabelPosition: "r" } },
    ], { x: gx, y: top + 0.28, w: gw, h: 3.3, catAxisLabelColor: K.gray700, valAxisLabelColor: K.gray700, catAxisLabelFontSize: 9, valAxisLabelFontSize: 8,
      catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", valGridLine: { color: K.rule, size: 0.5 }, catGridLine: { style: "none" },
      valAxisMinVal: 0, valAxisMaxVal: 300, valAxisMajorUnit: 50, valAxisLineShow: false, showLegend: false,
      dataLabelFontSize: 8, dataLabelFontFace: "+mn-lt", dataLabelColor: K.ink, objectName: "bep-chart" });
    const rx = 5.85, rw = 3.75;
    table(s, [
      ["百万円", "3年目", "4年目", "5年目"],
      ["販管費（固定費）", ...PR.slice(2).map((r) => String(mil(r.sga)))],
      ["粗利率", ...PR.slice(2).map((r) => Math.round(r.gpRate * 100) + "%")],
      ["損益分岐点売上", ...PR.slice(2).map((r) => String(mil(r.bep)))],
      ["実際の売上", ...PR.slice(2).map((r) => String(mil(r.sales)))],
      ["安全余裕率", ...PR.slice(2).map((r) => (r.sales > r.bep ? Math.round((1 - r.bep / r.sales) * 100) + "%" : "─"))],
    ], { x: rx, y: top, w: rw, colW: [1.5, 0.75, 0.75, 0.75], rowH: 0.32, size: 8.5, strongRows: [3], name: "bep-table" });
    box(s, { x: rx, y: top + 2.25, w: rw, h: 1.33, fill: K.ink, name: "bep-people" });
    txt(s, [
      run("人で見た損益分岐", { bold: true, fontSize: 10.5, breakLine: true }),
      run(`販管費／人 ${man(Y5.sga / Y5.staff)}万円 ÷ 粗利／拠点 ${man(U.gpYear)}万円 ＝ ${(Y5.sga / Y5.staff / U.gpYear).toFixed(1)}`, { breakLine: true }),
      run("新規は初年度半額 → 期末で1人約4拠点が目安", { bold: true, breakLine: true }),
      run(`4年目 ${PR[3].sitesPerStaff.toFixed(2)} → 5年目 ${Y5.sitesPerStaff.toFixed(2)}（まだ届かない）`),
    ], { x: rx + 0.15, y: top + 2.3, w: rw - 0.3, h: 1.25, size: 9, color: K.paper, psa: 2 });
    txt(s, `安全余裕率＝（売上−損益分岐点売上）÷売上。1〜2年目の損益分岐点売上は${mil(PR[0].bep)}・${mil(PR[1].bep)}百万円（売上${mil(PR[0].sales)}・${mil(PR[1].sales)}百万円）`, { x: X0, y: 4.98, w: W, h: 0.26, size: S.note, color: K.gray500 });
  }

  // ---------- Slide 10 資金計画・黒字倒産対策 ----------
  {
    const s = add(25);
    detailRef(s, DP.cash);
    heading(s, chap("④", `資金計画 ─ 自己資金${(V.GROWTH.funding.reduce((a, b) => a + b, 0) / 1e4 + 0.05).toFixed(2)}億円を、段階に分けて入れる`), `課題の前提「豊富な資金」。赤字の累計のピーク約${(-PLAN.cumPeak / 1e4).toFixed(1)}億円＋最低残高＋遅れへの備え`);
    const lx = X0, lw = 4.3, top = 1.15;
    txt(s, "資金の入れ方（すべて自己資金）", { x: lx, y: top, w: lw, h: 0.26, size: 10.5, bold: true });
    table(s, [
      ["時期", "入れる条件", "金額", "使い道"],
      ["準備期", "─", "500万円", "試作品・実証"],
      ["1年目", `実証で1台1日${be}分以上`, oku(V.GROWTH.funding[0]), "人員・開発"],
      ["3年目", "継続率90%・平均12台", oku(V.GROWTH.funding[2]), "人員・販売"],
      [{ text: "合計", bold: true }, "", { text: oku(V.GROWTH.funding.reduce((a, b) => a + b, 0) + 500), bold: true }, ""],
    ], { x: lx, y: top + 0.3, w: lw, colW: [0.65, 1.5, 1.15, 1.0], rowH: 0.36, size: 8.5, leftCols: [1, 3], strongRows: [4], name: "funding" });
    box(s, { x: lx, y: top + 2.25, w: lw, h: 1.45, fill: K.fill, name: "why-own" });
    txt(s, [run("なぜ一度に全部を入れないか", { bold: true, breakLine: true }), run("資金が豊富でも、線を越えたときだけ次を入れる。止まると決めたら残りは使わない。", { breakLine: true }),
      run("外の資金を入れるなら成熟期（上場）。融資・出資を使わないので、返済と利息はない", { color: K.gray700 })], { x: lx + 0.15, y: top + 2.25, w: lw - 0.3, h: 1.45, size: 9, valign: "middle", psa: 3 });

    const rx = 4.95, rw = 4.65;
    txt(s, "期末の現金残高（百万円）", { x: rx, y: top, w: rw, h: 0.26, size: 10.5, bold: true });
    txt(s, [run("━ 現金残高", { bold: true }), run("　┅ 最低残高（固定費3か月分）", { color: K.gray700 })], { x: rx + 1.6, y: top, w: rw - 1.6, h: 0.26, size: 8, align: "right", valign: "middle" });
    const lab = ["準備期", "1年目", "2年目", "3年目", "4年目", "5年目"];
    s.addChart([
      { type: pres.charts.LINE, data: [{ name: "現金残高", labels: lab, values: [0, ...PR.map((r) => mil(r.cash))] }], options: { chartColors: [K.ink], lineSize: 3, lineDataSymbol: "circle", lineDataSymbolSize: 6, showValue: true } },
      { type: pres.charts.LINE, data: [{ name: "最低残高", labels: lab, values: Array(6).fill(Math.round(PR[3].sga / 4 / 100)) }], options: { chartColors: [K.gray500], lineSize: 1.5, lineDash: "dash", lineDataSymbol: "none", showValue: false } },
    ], { x: rx, y: top + 0.28, w: rw, h: 2.2, catAxisLabelColor: K.gray700, valAxisLabelColor: K.gray700, catAxisLabelFontSize: 8.5, valAxisLabelFontSize: 8,
      catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", valGridLine: { color: K.rule, size: 0.5 }, catGridLine: { style: "none" },
      valAxisMinVal: 0, valAxisMaxVal: 175, valAxisMajorUnit: 25, valAxisLineShow: false, showLegend: false,
      dataLabelFontSize: 8, dataLabelFontFace: "+mn-lt", dataLabelColor: K.ink, dataLabelPosition: "t", objectName: "cash-chart" });

    box(s, { x: rx, y: top + 2.6, w: rw, h: 1.15, fill: K.ink, name: "anti-bankruptcy" });
    txt(s, "黒字倒産を防ぐ4つのルール", { x: rx + 0.15, y: top + 2.66, w: rw - 0.3, h: 0.26, size: 10.5, bold: true, color: K.paper });
    txt(s, [
      run("① 利用料は年額前払い。導入支援は着手金50%", { breakLine: true }),
      run("② 現金残高が固定費3か月分を下回りそうなら、先に採用を止める", { breakLine: true }),
      run("③ 月次の資金繰り表で、入金と支払いの時期を12か月先まで管理", { breakLine: true }),
      run("④ 3年目の追加資金は、継続率と台数の線を越えたときだけ入れる"),
    ], { x: rx + 0.15, y: top + 2.94, w: rw - 0.3, h: 0.78, size: 8.5, color: K.paper, psa: 1 });
    txt(s, `簡易計算（税・償却・運転資本を除く）。固定費3か月分＝4年目の販管費${mil(PR[3].sga)}百万円÷4。グラフの準備期は0から始めた`, { x: X0, y: 4.98, w: W, h: 0.3, size: S.note, color: K.gray500 });
  }

  // ---------- Slide ⑥黒字化の条件と撤退ライン ----------
  {
    const s = add(26);
    detailRef(s, DP.gates);
    heading(s, chap("④", "黒字化の条件 ─ 拠点数より、1拠点の台数"), "10台のままでは計画どおりでも5年で黒字にならない。15台に増やせれば新規が計画の6割でも5年目に黒字");
    L.gates(s, V);
    txt(s, "新規拠点・個社設定を同じ割合で減らし、社員数＝4＋（計画−4）×達成度で採用も遅らせた簡易計算。15台はAIグラス12台＋表示付き3台", { x: X0, y: 4.98, w: W, h: 0.26, size: S.note, color: K.gray500 });
  }

  // ---------- Slide ④年単位の計画 ----------
  {
    const s = add(27);
    detailRef(s, DP.pl);
    heading(s, chap("④", "年単位の計画 ─ 何をして、いくら売り、いくら残るか"), "事業の中身・拠点・社員・収支を1年ずつ。判定の線を越えたら次の年へ進む");
    const P = (f) => PR.map(f);
    table(s, [["", "準備期", ...PR.map((r) => r.year + "年目")],
      ["やること", "試作品・1拠点で実証", "試験導入3社。記録と報告を自動に", "本契約へ。写真台帳を各社の形式で", "図面との照合。大手1社を事例に", "表示付きグラスで手元AR（1工種）", "教材化。電線工場へ試験導入"],
      ["期末の拠点", "─", ...P((r) => String(r.sites))],
      ["社員", "1", ...P((r) => String(r.staff))],
      ["売上（百万円）", "─", ...P((r) => String(mil(r.sales)))],
      ["営業損益（百万円）", "▲5", ...P((r) => ({ text: (r.op < 0 ? "▲" : "") + Math.abs(mil(r.op)), color: r.op < 0 ? L.RED : K.ink, bold: true }))],
      ["期末の現金（百万円）", "─", ...P((r) => String(mil(r.cash)))],
      ["進む条件（判定の線）", `1台1日${be}分以上`, "2社以上が本契約", "平均12台", "継続率90%→追加資金", "─", "─"]],
      { x: X0, y: 1.12, w: W, colW: [1.5, 1.2, 1.3, 1.3, 1.3, 1.3, 1.3], rowH: [0.3, 0.62, 0.28, 0.28, 0.28, 0.28, 0.28, 0.4], size: 8.5, align: "center", leftCols: [0], strongRows: [5], name: "yearly" });
    box(s, { x: X0, y: 4.1, w: W, h: 0.78, fill: K.ink, name: "yearly-msg" });
    txt(s, [run("5年間は赤字。", { bold: true, color: L.YELLOW }), run(`5年目は売上約${mil(Y5.sales)}百万円・営業損益▲${mil(-Y5.op)}百万円で、あと約${gapSites}拠点分の粗利が足りない。`, { breakLine: true }),
      run("黒字化の条件：", { bold: true, color: L.YELLOW }), run(`効果が出た班から台数を増やし、1拠点15台にすると5年目に黒字（営業利益約${mil(D15.y5.op)}百万円・p.26）`)],
      { x: X0 + 0.15, y: 4.1, w: W - 0.3, h: 0.78, size: 9.5, color: K.paper, valign: "middle", psa: 2 });
    txt(s, "準備期の営業損益は準備費用500万円。1年目の3拠点は有償の試験導入（利用料なし）。簡易計算（税・償却・運転資本を除く）", { x: X0, y: 4.98, w: W, h: 0.26, size: S.note, color: K.gray500 });
  }

  // ---------- Slide ④10年で見ると ----------
  {
    const s = add(28);
    detailRef(s, "30");
    heading(s, chap("④", `10年で見ると ─ 10台でも${R10.firstBlack}年目に黒字、${R10.cumBack}年目に回収`), "5年目までの計画は変えずに、6〜10年目を同じ式で延ばした。条件は6年目から採用のペースを落とすこと");
    const yrs10 = R10.rows.map((r) => r.year + "年");
    const gx = X0, gw = 5.2, top = 1.12;
    txt(s, [run("━ 売上", { bold: true }), run("　■ 営業損益", { color: K.gray700 }), run("　┅ 累積の営業損益（百万円）", { color: K.gray700 })], { x: gx, y: top, w: gw, h: 0.24, size: 8, valign: "middle" });
    s.addChart([
      { type: pres.charts.LINE, data: [{ name: "売上", labels: yrs10, values: R10.rows.map((r) => mil(r.sales)) }], options: { chartColors: [K.ink], lineSize: 3, lineDataSymbol: "circle", lineDataSymbolSize: 5, showValue: false } },
      { type: pres.charts.LINE, data: [{ name: "営業損益", labels: yrs10, values: R10.rows.map((r) => mil(r.op)) }], options: { chartColors: [ACC], lineSize: 2.5, lineDataSymbol: "square", lineDataSymbolSize: 5, showValue: true } },
      { type: pres.charts.LINE, data: [{ name: "累積", labels: yrs10, values: R10.rows.map((r) => mil(r.cum)) }], options: { chartColors: [K.gray500], lineSize: 2, lineDash: "dash", lineDataSymbol: "none", showValue: false } },
    ], { x: gx, y: top + 0.26, w: gw, h: 2.55, catAxisLabelColor: K.gray700, valAxisLabelColor: K.gray700, catAxisLabelFontSize: 8, valAxisLabelFontSize: 8,
      catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", valGridLine: { color: K.rule, size: 0.5 }, catGridLine: { style: "none" },
      valAxisMinVal: -200, valAxisMaxVal: 700, valAxisMajorUnit: 100, valAxisLineShow: false, showLegend: false, catAxisLabelPos: "low",
      dataLabelFontSize: 7, dataLabelFontFace: "+mn-lt", dataLabelColor: ACC, dataLabelPosition: "t", objectName: "ten-year" });
    table(s, [["", "5年目", "6年目", "8年目", "10年目"],
      ["拠点", ...[4, 5, 7, 9].map((i) => String(R10.rows[i].sites))],
      ["社員", ...[4, 5, 7, 9].map((i) => String(R10.rows[i].staff))],
      ["社員1人あたり拠点", ...[4, 5, 7, 9].map((i) => R10.rows[i].sitesPerStaff.toFixed(1))]],
      { x: gx, y: top + 2.88, w: gw, colW: [1.6, 0.9, 0.9, 0.9, 0.9], rowH: 0.24, size: 8, name: "ten-kpi" });
    const rx = 5.85, rw = 3.75;
    txt(s, "前提を崩しても10年で黒字になるか", { x: rx, y: top, w: rw, h: 0.26, size: 10, bold: true });
    const yr = (v) => (v ? v + "年目" : "─");
    table(s, [["10年で", "黒字", "赤字を回収", "最低現金"],
      ...TEN.map((c, i) => [{ text: c.n, bold: i === 0 }, yr(c.fb), { text: yr(c.cb), color: c.cb ? L.GREEN : L.RED, bold: true }, { text: String(mil(c.min)), color: c.min < 0 ? L.RED : K.ink, bold: c.min < 0 }])],
      { x: rx, y: top + 0.3, w: rw, colW: [1.35, 0.7, 0.9, 0.8], rowH: 0.27, size: 8, align: "center", leftCols: [0], strongRows: [1], name: "ten-stress" });
    box(s, { x: rx, y: top + 2.32, w: rw, h: 1.48, fill: K.ink });
    txt(s, [run("ポイント", { bold: true, color: L.YELLOW, breakLine: true }),
      run(`5年で見ると赤字だが、積み上げ型なので6年目から黒字が厚くなる。10年目の営業利益は約${mil(R10.rows[9].op)}百万円（15台なら約${mil(R10_15.rows[9].op)}百万円）。`, { breakLine: true }),
      run("値下げとAI費用2倍では現金がいったんマイナスになり、自己資金の追加が要る", { color: L.YELLOW })], { x: rx + 0.12, y: top + 2.34, w: rw - 0.24, h: 1.44, size: 8.5, color: K.paper, valign: "middle", psa: 3 });
    txt(s, `6〜10年目の前提：新規 年36→40拠点（関東の中堅で頭打ち）、社員24→32人、解約は年10%。電線工場・海外は入れていない。10年目でターゲット市場の約${(R10.rows[9].sites * 300 / 141e2).toFixed(1)}%`, { x: X0, y: 4.98, w: W, h: 0.26, size: S.note, color: K.gray500 });
  }

  // ---------- Slide ⑤段階ごとの課題 ----------
  {
    const s = add(29);
    heading(s, chap("⑤", "段階ごとの課題 ─ 準備期・導入期・成長期・成熟期"), "各段階の一番の課題、起こりうること、打ち手、進む条件");
    const st = [
      ["準備期", "0〜6か月", "効果が出るか", "グラスの電池・録画の制約で記録が途切れる。削減が" + be + "分に届かない", "要所だけを記録。写真台帳から自動に。2〜3班で小さく測る", `1台1日${be}分以上`],
      ["導入期", "1〜2年目", "売れるか・定着するか", "年300万円を払うか未検証。現場が「監視」と感じて使わない。創業者が全部を抱える", "有償の試験導入。記録の範囲と労使の合意を先に。コンサル部隊の1人目を最初に採用", "1年目末に2社以上が本契約"],
      ["成長期", "3〜5年目", "台数と人", "10台のままでは黒字にならない。コンサル・エンジニアの採用が追いつかない。AI費用とラベル付けが増える", "効果が出た班から台数を増やす提案。採用を拠点数に連動。解析する場面に上限", "2年目末に平均12台・継続率90%"],
      ["成熟期", "6年目〜", "競合と次の市場", "施工管理アプリ・端末メーカーのAI化。データの持ち主の取り決め。工場・海外へ広げる体制", "電気工事の判断データと監修で差をつける。データ利用の契約を最初から。提携で広げる", "黒字の継続。上場は実績で判断"],
    ];
    const cw = (W - 3 * 0.1) / 4, top = 1.12;
    st.forEach(([ph, when, key, risk, act, gate], i) => {
      const x = X0 + i * (cw + 0.1);
      box(s, { x, y: top, w: cw, h: 0.62, fill: i === 0 ? ACC : i === 3 ? K.ink : K.gray700, name: "stage-" + i });
      txt(s, [run(ph, { bold: true, fontSize: 13 }), run("　" + when, { fontSize: 8.5, breakLine: true }), run("課題：" + key, { bold: true, fontSize: 10 })], { x: x + 0.1, y: top, w: cw - 0.2, h: 0.62, color: K.paper, valign: "middle" });
      box(s, { x, y: top + 0.68, w: cw, h: 1.3, fill: K.fill });
      txt(s, [run("起こりうること", { bold: true, fontSize: 8.5, color: ACC, breakLine: true }), run(risk, { fontSize: 8.5 })], { x: x + 0.1, y: top + 0.72, w: cw - 0.2, h: 1.22, valign: "top" });
      box(s, { x, y: top + 2.04, w: cw, h: 1.15, line: K.rule });
      txt(s, [run("打ち手", { bold: true, fontSize: 8.5, color: ACC, breakLine: true }), run(act, { fontSize: 8.5 })], { x: x + 0.1, y: top + 2.08, w: cw - 0.2, h: 1.08, valign: "top" });
      box(s, { x, y: top + 3.25, w: cw, h: 0.5, line: ACC, lw: 1.25 });
      txt(s, [run("進む条件：", { bold: true, fontSize: 8 }), run(gate, { fontSize: 8.5, bold: true })], { x: x + 0.1, y: top + 3.25, w: cw - 0.2, h: 0.5, valign: "middle" });
    });
    txt(s, "条件に届かなければ「直す」か「止まる」。線の全体は p.26。実現可能性の弱いところは次のページ", { x: X0, y: 4.98, w: W, h: 0.26, size: S.note, color: K.gray500 });
  }

  // ---------- Slide ⑤実現可能性の見直し ----------
  {
    const s = add(30);
    detailRef(s, "26・28・29");
    heading(s, chap("⑤", "実現可能性の見直し ─ 自分で見つけた弱いところ"), "計画を疑って見つけた点。隠さず、確かめ方・直し方とセットで示す");
    table(s, [["", "弱いところ", "何が問題か", "どう確かめる・直す"],
      ["1", "普段使いのグラスの性能", "一般向けAIグラスは動画1回3分程度まで（報道）。開発者向けの映像は720p・30fps。連続で撮ると電池が持たない", "要所の短い記録に絞る。準備期に実機を1日使って測る"],
      ["2", "現場の安全", "保護めがねの規格に合うカメラ付きグラスは少ない。落下・粉じんに弱い", "安全担当と確認。粉じん・高所は産業用（RealWear等）を選べる設計"],
      ["3", "データの通り道", "一般向けグラスはメーカーのアプリを経由する。図面・顧客情報の扱い", "規約と設定を確認。当社アプリ経由・国内保存に限る"],
      ["4", "値段", `年300万円は未検証。効果は支払いの約${CV.ratio.toFixed(1)}倍で余裕が小さい。試験導入で導入支援200万円も重い`, "試験導入3社に価格を聞く。写真台帳の自動化で削減分を上げる"],
      ["5", "黒字化", "1拠点10台のままでは5年で黒字にならない", "2年目末に平均12台を判定の線にする"],
      ["6", "人", `CTO（年収1,000万円）・コンサル${V.ROLES.find((r) => r[0] === "導入コンサルタント")[2][4]}人・ラベル付けの監修を採れるか`, "採用を拠点数に連動。顧問・大学と組む"],
      ["7", "獲得費用（CAC）", "1〜2年目の私の営業の時間をCACに入れていない", "入れると上がる。実績で更新して見せる"],
      ["8", "解約率10%", "日本の根拠がない（海外の目安）", "更新時の継続率を記録して差し替える"]],
      { x: X0, y: 1.1, w: W, colW: [0.3, 1.6, 4.0, 3.3], rowH: [0.26, 0.46, 0.4, 0.4, 0.46, 0.3, 0.4, 0.3, 0.3], size: 8, leftCols: [1, 2, 3], align: "center", name: "feasibility" });
    txt(s, "グラスの仕様は報道・メーカーの開発者向け資料（2024〜2025年）。国内の正規販売・技適・規格は要確認（詳細版 p.26）", { x: X0, y: 4.98, w: W, h: 0.26, size: S.note, color: K.gray500 });
  }

  // ---------- Slide 準備期の6か月 ----------
  {
    const s = add(31);
    heading(s, chap("⑤", "準備期の6か月 ─ 最初の課題「効果が出るか」を確かめる"), "費用は約500万円。自己資金でまかない、創業者の給料はゼロ");
    const lx = X0, lw = 4.4, top = 1.15;
    txt(s, "準備期の費用（万円）", { x: lx, y: top, w: lw, h: 0.26, size: 10.5, bold: true });
    const uses = [["試作品の開発", 250, "クラウド・生成AI・外注エンジニア"], ["グラス・機材", 80, "AIグラス・表示付きグラス・スマホ 数台"], ["1拠点での実証", 100, "3か月の交通・設置・現場研修"], ["設立・契約・知財", 40, "法人設立、秘密保持・データ利用契約"], ["予備", 30, ""]];
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
    [["3か月", "試作品：一人称の記録 → 写真の自動抽出 → 報告書の自動作成"], ["6か月", "1拠点で実証：報告書の作成時間と、グラスで1日記録できるかを実測"], ["6か月", "有償試験導入3拠点の契約、CTO・コンサルの採用開始"]].forEach(([t, v], i) => {
      const y = top + 0.32 + i * 0.5;
      box(s, { x: rx, y: y + 0.06, w: 0.75, h: 0.34, fill: K.ink });
      txt(s, t, { x: rx, y: y + 0.06, w: 0.75, h: 0.34, size: 9.5, bold: true, color: K.paper, align: "center", valign: "middle" });
      txt(s, v, { x: rx + 0.85, y, w: rw - 0.85, h: 0.46, size: 9.5, valign: "middle" });
    });
    box(s, { x: rx, y: top + 1.95, w: rw, h: 1.25, line: K.ink, lw: T.stroke.strong, name: "return" });
    txt(s, "準備期の後の資金", { x: rx + 0.15, y: top + 2.0, w: rw - 0.3, h: 0.28, size: 10.5, bold: true });
    txt(s, [
      run("1年目：", { bold: true }), run(`実証に合格したら、自己資金${oku(V.GROWTH.funding[0])}を入れる`, { breakLine: true }),
      run("3年目：", { bold: true }), run(`継続率と台数の線を越えたら${oku(V.GROWTH.funding[2])}を追加`, { breakLine: true }),
      run("お願い：", { bold: true }), run("実証先になってくれる電気工事会社のご紹介"),
    ], { x: rx + 0.15, y: top + 2.3, w: rw - 0.3, h: 0.86, size: 9, psa: 3 });
    txt(s, `実証の合格ラインは1台1日${be}分の削減（詳細版 p.8）。グラスの電池と録画の制約は実機で確認（詳細版 p.26）`, { x: X0, y: 5.0, w: W, h: 0.24, size: S.note, color: K.gray500 });
  }

  // ---------- Slide 12 リスクと覚悟 ----------
  {
    const s = add(32);
    heading(s, chap("⑤", "リスク ─ 最大のリスクは自分。役割と判断基準を先に決める"), "弱みは隠さない。対策とセットで示す");
    const risks = [
      ["創業者がボトルネックになる", "役割を「型を渡す」に絞る。BSCの4指標で判断を任せ、1か月不在でも回る会社にする"],
      ["前職との関係", "前職の顧客・機密は使わない。退職時の合意（競業・守秘）を確認し、紹介は第三者経由で"],
      ["競合のAI化", "施工管理アプリ・RealWear等とは競わずにつなぐ。判断の蓄積と施工経験者の監修を障壁にする"],
      ["常時収録への抵抗", "作業時間・作業エリアに限定。本人同意と労使合意、休憩中・私語は自動除外"],
      ["資金の使いすぎ", "自己資金でも段階に分けて入れる。最低残高ルールと前払い・着手金"],
    ];
    risks.forEach(([r, a], i) => {
      const y = 1.15 + i * 0.5;
      hline(s, X0, y, W, K.rule, T.stroke.rule);
      txt(s, r, { x: X0, y, w: 2.4, h: 0.5, size: 10.5, bold: true, valign: "middle" });
      txt(s, "→ " + a, { x: X0 + 2.45, y, w: W - 2.45, h: 0.5, size: 9.5, valign: "middle" });
    });
    hline(s, X0, 1.15 + 5 * 0.5, W, K.rule, T.stroke.rule);
    box(s, { x: X0, y: 3.88, w: W, h: 1.2, fill: K.ink, name: "closing" });
    txt(s, "対策の共通点：弱みは先に言い、止まる線を先に引く", { x: X0 + 0.2, y: 3.98, w: W - 0.4, h: 0.4, size: 15, bold: true, color: K.paper });
    txt(s, "進むかどうかは、思いではなく数字で決める。そのうえで、続けたい理由は次のページ。", { x: X0 + 0.2, y: 4.45, w: W - 0.4, h: 0.5, size: 12, color: K.paper });
  }

  // ---------- Slide 最後に（創業者の願い） ----------
  {
    const s = add(33);
    heading(s, "最後に ─ 事業への思い", "黒字は目的ではなく、続けるための条件。社会の役に立つ事業として、見返りを求めずに始める");
    const top = 1.12, lw = 3.6;
    box(s, { x: X0, y: top, w: lw, h: 2.75, fill: K.fill, name: "impact" });
    txt(s, "10年で現場に残るもの（計算）", { x: X0 + 0.15, y: top + 0.08, w: lw - 0.3, h: 0.26, size: 10.5, bold: true });
    [[`約${man(Math.round(IMPACT.hours / 1e4))}万時間`, `書類に使っていた時間が、現場と若手の指導に戻る（約${man(Math.round(IMPACT.hours / V.WAGE.annualHours / 10) * 10)}人分の1年）`],
      [`約${Math.round(IMPACT.value / 1e4)}億円`, "その時間の人件費の価値（書類の分だけ。手戻りは入れていない）"],
      [`延べ${man(IMPACT.siteYears)}拠点年`, "熟練者の判断が記録され、教材として次の担い手に渡る"]].forEach(([n, d], i) => {
      const y = top + 0.42 + i * 0.76;
      txt(s, n, { x: X0 + 0.15, y, w: lw - 0.3, h: 0.36, size: 17, bold: true, color: ACC, valign: "middle" });
      txt(s, d, { x: X0 + 0.15, y: y + 0.36, w: lw - 0.3, h: 0.36, size: 8.5, color: K.gray700 });
    });
    const rx = X0 + lw + 0.25, rw = W - lw - 0.25;
    txt(s, "創業者としての姿勢（社会起業家として）", { x: rx, y: top, w: rw, h: 0.26, size: 10.5, bold: true });
    [["利益は目的ではなく、続けるための条件", "配当は求めない。利益は現場の支援と開発に戻す"],
      ["自己資金は、見返りを求めない投資", "クラウドファンディングのように、共感する工事会社・職人・業界団体と一緒に育てる"],
      ["技能の教材は、業界の財産に", "記録から作った教材の一部を、業界団体と学校に無償で公開する"],
      ["それでも、止まる線は守る", "資金は有限。撤退ラインを守り、続けられる形を探し続けることが責任"]].forEach(([h, d], i) => {
      const y = top + 0.32 + i * 0.6;
      box(s, { x: rx, y: y + 0.05, w: 0.36, h: 0.42, fill: i === 3 ? K.gray700 : ACC });
      txt(s, String(i + 1), { x: rx, y: y + 0.05, w: 0.36, h: 0.42, size: 12, bold: true, color: K.paper, align: "center", valign: "middle" });
      txt(s, [run(h, { bold: true, breakLine: true }), run(d, { fontSize: 8.5, color: K.gray700 })], { x: rx + 0.45, y, w: rw - 0.45, h: 0.55, size: 10, valign: "middle" });
    });
    box(s, { x: X0, y: 4.0, w: W, h: 0.88, fill: K.ink, name: "wish" });
    txt(s, [run("日本の品質を支えてきた現場の勘を、人に依存しない形で次の担い手へ。", { bold: true, fontSize: 14, breakLine: true }),
      run("私の資金と時間は、そのために使います。最初の目標は、6か月で最初の1拠点を動かすことです。", { fontSize: 11 })],
      { x: X0 + 0.2, y: 4.0, w: W - 0.4, h: 0.88, color: K.paper, valign: "middle", psa: 3 });
    txt(s, "ご清聴ありがとうございました", { x: 3.85, y: 5.21, w: 2.65, h: 0.28, size: 10.5, bold: true, color: K.ink, align: "right", valign: "middle" });
    txt(s, "時間と価値は10年計画（10台）の拠点数×1拠点の削減（年1,400時間・約500万円）から。仮説の削減時間（1台1日35分）が前提で、実証で測る", { x: X0, y: 4.98, w: W, h: 0.26, size: S.note, color: K.gray500 });
  }

  fs.mkdirSync(OUT_DIR, { recursive: true });
  await pres.writeFile({ fileName: OUT });
  console.log("wrote", OUT, "台本の合計", NOTES.total);
}

main().catch((e) => { console.error(e); process.exit(1); });
