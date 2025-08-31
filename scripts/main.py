# -*- coding: utf-8 -*-
"""
Generate daily AI commute podcast MP3 + RSS.
強化ポイント:
- OPENAI_API_KEYの妥当性チェック（*** 等の誤値で落ちる問題を事前検出）
- TZ付き日時で RSS pubDate/lastBuildDate を生成（ValueError対策）
- feed 必須フィールド（title/link/description）を常にセット
- TTS失敗時は自動リトライ＆スキップ、ゼロ件ならフォールバック音声を生成
- base_url を Secrets or 自動推定（余計なスペース除去）
- mp3 length を実ファイルサイズで設定
- 取得記事が薄いときも LLM で拡張（技術濃度/ユーモア）
"""

import os
import io
import re
import sys
import time
import math
import json
import random
import textwrap
import traceback
import datetime as dt
from typing import List, Dict, Any

import yaml
import httpx
import feedparser
from feedgen.feed import FeedGenerator
from pydub import AudioSegment

# --------- util ---------

def log(msg: str):
    print(msg, flush=True)

def utcnow():
    return dt.datetime.now(dt.timezone.utc)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def sanitize_base_url(url: str | None) -> str | None:
    if not url:
        return None
    url = url.strip()
    # 余計なスペース混入の事故対策
    url = re.sub(r"\s+", "", url)
    return url

def infer_pages_base_url() -> str | None:
    """
    FEED_BASE_URL が未指定の場合に github.repository から推定
    例: seitoshiki/-ai-commute-audio- -> https://seitoshiki.github.io/-ai-commute-audio-
    """
    repo = os.getenv("GITHUB_REPOSITORY")  # owner/repo
    if not repo or "/" not in repo:
        return None
    owner, repo_name = repo.split("/", 1)
    return f"https://{owner.lower()}.github.io/{repo_name}"

def validate_openai_api_key():
    key = os.getenv("OPENAI_API_KEY", "").strip()
    # GitHub の誤設定で "***" が入ると httpx が "Illegal header value b'***'" で落ちる
    if not key or set(key) == {"*"} or key == "***":
        raise RuntimeError(
            "OPENAI_API_KEY が正しく設定されていません。"
            "GitHub の Secrets に有効な API キー（例: sk- で始まる）を保存し、"
            "Actions の再実行をしてください。"
        )
    # 新形式でも受け入れるが、露骨に短い/不正は弾く
    if len(key) < 20:
        raise RuntimeError("OPENAI_API_KEY が不正な形式です。")

# --------- news fetch ---------

def google_news_rss_url(query: str, lang="ja", country="JP") -> str:
    # "when:1d" のような修飾子は Google News の q= にそのまま含めてOK
    q = httpx.QueryParams({"q": query, "hl": lang, "gl": country, "ceid": f"{country}:{lang}"})
    return f"https://news.google.com/rss/search?{q}"

def fetch_items_from_config(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    # Google News
    for q in cfg["sources"].get("google_news_queries", []):
        url = google_news_rss_url(q)
        feed = feedparser.parse(url)
        for e in feed.entries[:20]:  # 各クエリにつき上位20件
            items.append({
                "title": e.get("title", ""),
                "summary": e.get("summary", ""),
                "link": e.get("link", ""),
                "published": e.get("published", ""),
            })
    # extra RSS
    for rss in cfg["sources"].get("extra_rss", []):
        feed = feedparser.parse(rss)
        for e in feed.entries[:20]:
            items.append({
                "title": e.get("title", ""),
                "summary": e.get("summary", ""),
                "link": e.get("link", ""),
                "published": e.get("published", ""),
            })
    log(f"[INFO] fetched items: {len(items)}")
    return items

def filter_items(items: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    inc = set(cfg["filters"].get("must_include_any", []))
    exc = set(cfg["filters"].get("must_exclude_any", []))
    def ok(it):
        title = (it.get("title") or "") + " " + (it.get("summary") or "")
        # include
        if inc and not any(k.lower() in title.lower() for k in inc):
            return False
        # exclude
        if any(k.lower() in title.lower() for k in exc):
            return False
        return True
    out = [it for it in items if ok(it)]
    log(f"[INFO] filtered items: {len(out)}")
    return out

def pick_for_duration(items: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    random.shuffle(items)
    target_min = cfg["schedule"]["target_total_minutes"]
    per_min = (cfg["schedule"]["per_item_seconds_min"] + cfg["schedule"]["per_item_seconds_max"]) / 2 / 60
    max_items = cfg["schedule"]["max_items"]
    n = min(max_items, max(5, int(target_min / per_min)))
    picked = items[:n]
    log(f"[INFO] picked items: {len(picked)}")
    return picked

# --------- LLM summary ---------

def openai_client():
    from openai import OpenAI
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def build_prompt_style(cfg: Dict[str, Any]) -> str:
    extra_topics = textwrap.dedent("""
    追加トピックを適宜織り込む（最新海外AIニュース、松尾研究室、落合陽一、プロンプト・マルチエージェント、自律エージェント、論文、ビジネス活用、GitHub活用、深津式プロンプトの学び、簿記3級、経理転職、基本情報技術者、筋トレ）。
    ただしURL・記号の読み上げは禁止。略語を一文字ずつ読むのも禁止（例：LLMは「エルエルエム」ではなく“エルエルエム”の単語として普通に）。
    """).strip()
    return f"{cfg['summary']['style']}\n{extra_topics}"

SYSTEM_PROMPT = """あなたは日本語のオーディオ台本作成アシスタントです。
出力は以下2部構成：
(1) 一人語り：導入→ニュースの要点→背景→技術の肝→現実的な活用→オチの一言。カジュアルでテンポよく、比喩を少なく、内容は濃く。
(2) 二人語り：司会と相棒の短い掛け合い（最大30秒）。相棒は時々ツッコミやユーモア。
禁止事項：URL、記号列の読み上げ、出典の羅列、長すぎる前置き。
"""

def summarize_with_openai(items: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    client = openai_client()
    style = build_prompt_style(cfg)
    out: List[Dict[str, Any]] = []
    for idx, it in enumerate(items, 1):
        base = f"タイトル: {it.get('title','')}\n要約: {it.get('summary','')}\n"
        user = f"{base}\nこの話題について、{style}\n文字数:{cfg['summary']['max_chars_per_item']}字以内で。"
        # リトライ
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=cfg["summary"]["temperature"],
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user}
                    ],
                )
                text = resp.choices[0].message.content.strip()
                # URLや余計な括弧の除去
                text = re.sub(r"https?://\S+", "", text)
                text = text.replace("()", "")
                out.append({
                    "title": it.get("title",""),
                    "text": text
                })
                break
            except Exception as e:
                log(f"[WARN] LLM summarize failed {idx} try={attempt+1}: {e}")
                time.sleep(2 * (attempt + 1))
        else:
            # 失敗時フォールバック
            out.append({
                "title": it.get("title",""),
                "text": f"話題「{it.get('title','')}」について：要点、背景、技術、影響を簡潔に紹介します。"
            })
    log(f"[INFO] texts_for_tts: {len(out)}")
    return out

# --------- TTS ---------

def tts_openai(text: str, voice: str, model: str, speaking_rate: float, bitrate_kbps: int) -> AudioSegment:
    """
    OpenAI TTS（非ストリーミング）。接続エラーは数回リトライ。
    """
    from openai import APIConnectionError
    client = openai_client()

    # 長文をチャンクに分割
    chunk_size = 700
    chunks = []
    buf = ""
    for line in text.splitlines():
        if len(buf) + len(line) + 1 > chunk_size:
            if buf:
                chunks.append(buf)
                buf = ""
        buf += (line + " ")
    if buf:
        chunks.append(buf)

    segments: List[AudioSegment] = []
    for i, ch in enumerate(chunks, 1):
        for attempt in range(3):
            try:
                audio = client.audio.speech.create(
                    model=model,
                    voice=voice,
                    input=ch,
                    format="mp3",
                    speed=speaking_rate,
                )
                mp3_bytes = audio.content  # SDK v1.44 以降
                seg = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
                segments.append(seg)
                break
            except APIConnectionError as e:
                log(f"[WARN] TTS connection error (retry) part={i}: {e}")
                time.sleep(2 * (attempt + 1))
            except Exception as e:
                log(f"[WARN] TTS failed part={i}: {e}")
                time.sleep(1)
                break  # 次のチャンクへ（このチャンクは諦める）

    if not segments:
        raise RuntimeError("TTSがすべて失敗しました。APIキーやネットワーク設定を確認してください。")

    out = segments[0]
    for s in segments[1:]:
        out += s
    return out

# --------- Build program ---------

def build_program_from_texts(texts: List[Dict[str, Any]], cfg: Dict[str, Any]) -> AudioSegment:
    voice = cfg["tts"]["voice"]
    model = cfg["tts"]["model"]
    rate = float(cfg["tts"]["speaking_rate"])
    bitrate_k = int(cfg["tts"]["output_bitrate"])

    intro_text = "通勤AIニュース。今日も技術濃度高めでいきます。"
    outro_text = "以上、通勤AIニュースでした。良い一日を！"

    program = AudioSegment.silent(duration=300)  # 0.3秒の無音で開始ブツ切れ防止
    try:
        program += tts_openai(intro_text, voice, model, rate, bitrate_k)
    except Exception as e:
        log(f"[WARN] intro TTS failed: {e}")

    # 本編
    item_ok = 0
    for i, t in enumerate(texts, 1):
        try:
            program += AudioSegment.silent(duration=200)
            head = f"トピック {i}。{t['title']}。"
            body = t["text"]
            seg = tts_openai(head + "\n" + body, voice, model, rate, bitrate_k)
            program += seg
            item_ok += 1
        except Exception as e:
            log(f"[WARN] item TTS failed {i}: {e}")

    try:
        program += AudioSegment.silent(duration=200)
        program += tts_openai(outro_text, voice, model, rate, bitrate_k)
    except Exception as e:
        log(f"[WARN] outro TTS failed: {e}")

    # 全滅フォールバック
    if item_ok == 0:
        log("[WARN] no items succeeded; building fallback audio")
        fallback = "今日は技術的な問題で短縮版です。明日はもっと面白く、濃い話をお届けします。"
        program = tts_openai(fallback, voice, model, rate, bitrate_k)

    return program

# --------- RSS ---------

def build_and_save_podcast(mp3_path: str, out_dir: str, base_url: str, site_title: str, site_author: str, site_desc: str):
    fg = FeedGenerator()
    fg.title(site_title)
    fg.link(href=base_url, rel='alternate')
    fg.language('ja')
    fg.description(site_desc)
    now = utcnow()
    fg.pubDate(now)
    fg.lastBuildDate(now)
    fg.generator('python-feedgen')

    # item
    title = f"{site_title} {dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).date()}"
    fe = fg.add_entry()
    fe.title(title)
    fe.description("本日の主要AIニュースを、日本語でやさしく要点解説。")
    rel_mp3 = os.path.basename(mp3_path)
    mp3_url = f"{base_url.rstrip('/')}/{rel_mp3}"
    fe.guid(mp3_url, permalink=False)
    size = os.path.getsize(mp3_path) if os.path.exists(mp3_path) else 0
    fe.enclosure(mp3_url, str(size), 'audio/mpeg')
    fe.pubDate(now)

    # write feed
    ensure_dir(out_dir)
    xml = fg.rss_str(pretty=True)
    with open(os.path.join(out_dir, "feed.xml"), "wb") as f:
        f.write(xml)

# --------- main ---------

def main():
    validate_openai_api_key()

    cfg = read_yaml("config.yaml")

    # base_url決定（優先度：config.yaml > Secrets(FEED_BASE_URL) > 自動推定）
    base_url = sanitize_base_url(cfg["site"].get("base_url")) or \
               sanitize_base_url(os.getenv("FEED_BASE_URL")) or \
               sanitize_base_url(infer_pages_base_url())
    if not base_url:
        raise RuntimeError("base_url を決定できませんでした。config.yaml か Secrets:FEED_BASE_URL を設定してください。")
    log(f"[INFO] base_url = {base_url}")

    # 記事収集→フィルタ→選定
    items = fetch_items_from_config(cfg)
    items = filter_items(items, cfg)
    items = pick_for_duration(items, cfg)

    # LLM要約（技術濃度＆ユーモア）
    texts = summarize_with_openai(items, cfg)

    # 音声合成
    out_dir = "output"
    ensure_dir(out_dir)
    today = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d")
    mp3_name = f"{today}-ai-commute.mp3"
    mp3_path = os.path.join(out_dir, mp3_name)

    program = build_program_from_texts(texts, cfg)
    # 注意: export時のbitrateはpydubの引数で指定（_thing_）
    program.export(mp3_path, format="mp3", bitrate=f"{cfg['tts']['output_bitrate']}k")
    log(f"[INFO] mp3 saved: {mp3_path} size={os.path.getsize(mp3_path)}")

    # RSS生成
    build_and_save_podcast(
        mp3_path=mp3_path,
        out_dir=out_dir,
        base_url=base_url,
        site_title=cfg["site"]["title"],
        site_author=cfg["site"]["author"],
        site_desc=cfg["site"].get("description","通勤向け：AI関連ニュースの音声ダイジェスト（日本語）"),
    )
    log("[INFO] feed.xml generated")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("[ERROR] " + str(e))
        traceback.print_exc()
        sys.exit(1)
