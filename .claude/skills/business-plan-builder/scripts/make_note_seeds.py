#!/usr/bin/env python3
"""事業計画の要約（plan_summary.json）から、note記事の種をつくる。

使い方:
  python3 make_note_seeds.py plan_summary.json <出力フォルダ> [--date YYYY-MM-DD]

出力:
  <出力フォルダ>/topics.yaml        記事の企画の一覧（topic・angle・article_type・素材・公開前の確認）
  <出力フォルダ>/materials/*.md     企画ごとの素材。frontmatter は note-article-generator の形式
                                    （date・topic・angle・article_type・model・status・references・note）

素材は「結論 → 具体的には（番号付き）→ 使える数字 → 自分の経験とのつなぎ → 公開前の確認 → 次に検討したい事業の切り口」の順。
記事の本文は書かない。本文は note-article-generator に「この素材で記事タイプ A を書いて」と渡して書く。
plan_summary.json の項目は references/note_bridge.md を参照。無い項目を使う企画は出さない。
"""
import datetime as dt
import json
import sys
from pathlib import Path


def q(s) -> str:
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def man(v) -> str:
    """万円 → 読みやすい表記"""
    v = round(v)
    sign = "▲" if v < 0 else ""
    v = abs(v)
    if v >= 10000:
        return f"{sign}{v / 10000:.1f}億円"
    return f"{sign}{v:,}万円"


def year(v) -> str:
    return f"{v}年目" if v else "期間内にしない"


def seeds(S: dict) -> list:
    biz, cust = S.get("business", ""), S.get("customer", "")
    e, u, p5, p10 = S.get("effect", {}), S.get("unit", {}), S.get("plan5", {}), S.get("plan10", {})
    out = []

    if S.get("lessons"):
        L = S["lessons"]
        out.append(dict(
            slug="numbers", type="A",
            topic="事業計画の数字に根拠を持たせる",
            angle=f"事業計画の数字を作って分かった、根拠の出し方{min(len(L), 4)}つ",
            conclusion="事業計画の数字は、統計と公表価格から積み上げ、仮説は仮説として分けて置くと、質問されても崩れない。",
            points=L[:4],
            numbers=[x for x in [
                e.get("yen") and f"お客様の効果 年{man(e['yen'])}（支払いの{e.get('ratio')}倍）",
                e.get("breakEvenMin") and f"元が取れる条件：1台1日{e['breakEvenMin']}分",
                u.get("grossProfit") and f"1拠点の粗利 年{man(u['grossProfit'])}（粗利率{u.get('margin')}%）",
            ] if x],
            link="現場で書類の時間を測ってきた経験と、統計の原表を自分で当たった話につなげる。",
        ))

    if p5 or p10:
        cond = S.get("blackCondition", "")
        out.append(dict(
            slug="honest-red", type="A",
            topic="赤字の計画をそのまま出す",
            angle="5年で黒字にならない計画を、そのまま出した理由",
            conclusion="黒字になる値を逆算して入れず、赤字のまま出して「何が変われば黒字になるか」を条件で示した。そのほうが質問に強い。",
            points=[x for x in [
                "黒字になる値を逆算すると、どの前提を盛ったのか自分でも説明できなくなる",
                cond and f"黒字化の条件を1つに絞った：{cond}",
                "5年で切らずに10年まで同じ式で延ばすと、利用料が積み上がる事業の姿が見える",
                "撤退ライン（進む・直す・止まる）を先に決めておくと、赤字でも続ける理由と止める理由がはっきりする",
            ] if x],
            numbers=[x for x in [
                p5 and f"5年目：{p5.get('sites', '─')}拠点・売上{p5.get('salesOku', '─')}億円・利益{man(p5.get('op5', 0))}・" + (f"{p5['black']}年目に黒字" if p5.get("black") else "5年では黒字にならない"),
                p10 and f"10年：黒字化{year(p10.get('black'))}・累計の赤字の回収{year(p10.get('cumBack'))}・10年目の利益{man(p10.get('op10', 0))}",
            ] if x],
            link="「融資のためによく見せたいのか」と問われて、計画を見直した場面を入れる。",
        ))

    if S.get("assumptions"):
        A = S["assumptions"]
        out.append(dict(
            slug="assumptions", type="B",
            topic="事業計画の前提の仕分け",
            angle=f"事業計画の前提を全部並べる：統計・公表・見積・仮説の4つで仕分けた（{len(A)}項目）",
            conclusion="前提を確度で4つに分けると、どこを実証で測ればよいかが一覧で分かる。",
            points=[f"{c}（{sum(1 for a in A if a[2] == c)}件）：" + "、".join(f"{a[0]} {a[1]}" for a in A if a[2] == c)
                    for c in ["統計", "公表", "見積", "仮説"] if any(a[2] == c for a in A)],
            numbers=[f"仮説の項目 {sum(1 for a in A if a[2] == '仮説')}件／全{len(A)}件"],
            link="カテゴリーごとに「> 感想 (後日追記):」を置き、実証で測れたら書き足す育てる記事にする。",
        ))

    if S.get("competitor"):
        out.append(dict(
            slug="big-competitor", type="A",
            topic="大手と同じ方向の事業の考え方",
            angle="大手が同じ方向の事業をしていると分かったときに、最初にやった3つのこと",
            conclusion="大手を競合から外さずに比べ、稼ぎ方を調べて、領域をずらした。",
            points=[
                f"同じ方向の大手を先に探す（例：{S['competitor']}）。人が入力するアプリだけを競合にしない",
                "大手の稼ぎ方を調べる：AIそのものを売っているのか、既存の保守契約に組み込んでいるのか",
                f"領域をずらす：{S.get('position', '大手の得意な領域ではなく、届いていないお客様に絞る')}",
            ],
            numbers=[],
            link="公表資料の範囲で書き、他社名には出典を付ける。",
        ))

    if S.get("weakPoints"):
        W = S["weakPoints"]
        out.append(dict(
            slug="weak-points", type="A",
            topic="計画の弱いところを先に言う",
            angle=f"自分の事業計画の弱いところを{len(W)}つ、先に書き出した",
            conclusion="弱いところは隠すより先に言い、確かめ方とセットにしたほうが信用される。",
            points=W,
            numbers=[],
            link="現場の安全の考え方（危ないところを先に洗い出す）とつなげる。",
        ))

    if S.get("making"):
        M = S["making"]
        out.append(dict(
            slug="making", type="D",
            topic="事業計画を資料にするまで",
            angle=M.get("angle", "研修の事業計画を10分・16枚にまとめるまで"),
            conclusion=M.get("conclusion", "詳しい本編をつくってから16枚に統合すると、削っても中身が残る。"),
            points=M.get("steps", []),
            numbers=M.get("numbers", []),
            link=M.get("link", "AIに手伝わせた部分と、自分の言葉に直した部分を分けて書く。"),
        ))
    return out


def material(seed: dict, S: dict, date: str) -> str:
    fm = [
        "---",
        f"date: {date}",
        f"topic: {q(seed['topic'])}",
        f"angle: {q(seed['angle'])}",
        f"article_type: {seed['type']}",
        "model: ",
        "status: draft",
    ]
    refs = S.get("references", [])
    fm += ["references:", *[f"  - {r}" for r in refs]] if refs else ["references: []"]
    fm += [
        "note:",
        f"  - {q('事業計画（' + S.get('slug', '') + '）から作った素材。本文は note-article-generator で書く')}",
        f"  - {q('職歴と結びつく内容。公開前にペンネームの匿名性を確かめる')}",
        "---",
        "",
    ]
    body = [
        f"# {seed['angle']}",
        "",
        "## 結論",
        seed["conclusion"],
        "",
        "## 具体的には",
        *[f"{i}. {p}" for i, p in enumerate(seed["points"], 1)],
        "",
    ]
    if seed["numbers"]:
        body += ["## 使える数字", *[f"- {n}" for n in seed["numbers"]], ""]
    body += [
        "## 自分の経験とのつなぎ",
        seed["link"],
        "",
        "## 事業の中身（素材）",
        f"- 事業：{S.get('business', '')}",
        f"- お客様：{S.get('customer', '')}",
        f"- 値段：{S.get('price', '')}",
        "",
        "## 公開前の確認",
        "- 発表者名・社内の情報・前職の顧客を書いていないか",
        "- 勤務先の業界・電圧などの仕様・資格の組み合わせで、本人が特定されないか",
        "- 研修の提出前に事業のアイデアを公開してよいか",
        "- 他社名は公表資料の範囲で、出典を付けたか",
        "",
        "## 次に検討したい事業の切り口",
        *([f"- {x}" for x in S.get("ideas", [])] or ["- （記事を書きながら1行残す）"]),
        "",
    ]
    return "\n".join(fm + body)


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    S = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = Path(sys.argv[2])
    date = sys.argv[sys.argv.index("--date") + 1] if "--date" in sys.argv else dt.date.today().isoformat()
    (out / "materials").mkdir(parents=True, exist_ok=True)
    slug = S.get("slug", "plan")
    items = seeds(S)
    lines = [f"# {slug} の事業計画から作った記事の企画（{date}）", "topics:"]
    for s in items:
        f = f"materials/{slug}-{s['slug']}.md"
        (out / f).write_text(material(s, S, date), encoding="utf-8")
        lines += [
            f"  - topic: {q(s['topic'])}",
            f"    angle: {q(s['angle'])}",
            f"    article_type: {s['type']}",
            f"    material: {f}",
            "    status: idea",
            f"    publish_check: {q('匿名性・研修の提出時期・他社名の出典を確かめてから公開')}",
        ]
    (out / "topics.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(items)}本の企画を書き出しました → {out}")
    for s in items:
        print(f"  [{s['type']}] {s['angle']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
