# -*- coding: utf-8 -*-
"""
Generate daily AI commute podcast MP3 + RSS.

今回の修正点:
- FEED_BASE_URL が "***" 等のダミーなら自動推定に切替
- OpenAI Chat/TTS が接続エラーでも進行（フォールバック台本＋ローカルTTS）
- OpenAI TTS: SDK差分 (format / response_format) に両対応
- 取得ゼロ/短文でも番組が無音にならない安全策
- RSS の日時は timezone 付き
"""

import os
import io
import re
import sys
import time
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
import subprocess
import tempfile
import uuid

# ---------- utils ----------

def log(msg: str):
    print(msg, flush=True)

def utcnow():
    return dt.datetime.now(dt.timezone.utc)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def is_placeholder(val: str | None) -> bool:
    if not val:
        return False
    v = val.strip()
    return (set(v) == {"*"} or v == "***" or v.lower().startswith("changeme"))

def sanitize_base_url(url: str | None) -> str | None:
    if not url:
        return None
    url = re.sub(r"\s+", "", url.strip())
    if is_placeholder(url):
        return None
    return url

def infer_pages_base_url() -> str | None:
    repo = os.getenv("GITHUB_REPOSITORY")  # owner/repo
    if not repo or "/" not in repo:
        return None
    owner, repo_name = repo.split("/", 1)
    return f"https://{owner.lower()}.github.io/{repo_name}"

def validate_openai_api_key():
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        log("[WARN] OPENAI_API_KEY が未設定。LLM/TTSはフォールバックで実行します。")
        return
    if is_placeholder(key):
        log("[WARN] OPENAI_API_KEY がプレースホルダーっぽい値です。フォールバックを使用します。")
        os.environ["OPENAI_API_KEY"] = ""  # 無効化
        return
    if len(key) < 20:
        log("[WARN] OPENAI_API_KEY の長さが短すぎます。フォールバックを使用します。")
        os.environ["OPENAI_API_KEY"] = ""

# ---------- fetch news ----------

def google_news_rss_url(query: str, lang="ja", country="JP") -> str:
    q = httpx.QueryParams({"q": query, "hl": lang, "gl": country, "ceid": f"{country}:{lang}"})
    return f"https://news.google.com/rss/search?{q}"

def fetch_items_from_config(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for q in cfg["sources"].get("google_news_queries", []):
        url = google_news_rss_url(q)
        feed = feedparser.parse(url)
        for e in feed.entries[:20]:
            items.append({
                "title": e.get("title", ""),
                "summary": e.get("summary", ""),
                "link": e.get("link", ""),
                "published": e.get("published", ""),
            })
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
        if inc and not any(k.lower() in title.lower() for k in inc):
            return False
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

# ---------- LLM summary (with fallback) ----------

def openai_client():
    from openai import OpenAI
    # タイムアウトを明示
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY") or None, timeout=60)

SYSTEM_PROMPT = """あなたは日本語のオーディオ台本作成アシスタントです。
出力は2部構成：
(1) 一人語り：導入→要点→背景→技術の肝→実務活用→一言。
(2) 二人語り：司会と相棒の軽い掛け合い（最大30秒）。
禁止：URL、記号列の読み上げ、出典の羅列、冗長な前置き。
砕けたトーンだが内容は濃く、ユーモアは控えめにキメる。
"""

def build_prompt_style(cfg: Dict[str, Any]) -> str:
    extra = textwrap.dedent("""
    追加トピックを自然に織り込む（海外AI動向、松尾研究室、落合陽一、プロンプト／マルチエージェント、自律エージェント、論文、ビジネス活用、GitHub活用、深津式プロンプト、簿記3級、経理転職、基本情報技術者、筋トレ）。
    URLや記号列は絶対に読まない。略語は一文字ずつ読まず単語として自然に。
    """).strip()
    return f"{cfg['summary']['style']}\n{extra}"

def summarize_with_openai(items: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    use_openai = bool(os.getenv("OPENAI_API_KEY"))
    client = openai_client() if use_openai else None
    style = build_prompt_style(cfg)

    for idx, it in enumerate(items, 1):
        base = f"タイトル: {it.get('title','')}\n要約: {it.get('summary','')}\n"
        user = f"{base}\nこの話題について、{style}\n文字数:{cfg['summary']['max_chars_per_item']}字以内で。"
        text: str | None = None

        if use_openai:
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
                    t = resp.choices[0].message.content.strip()
                    t = re.sub(r"https?://\S+", "", t)
                    t = t.replace("()", "")
                    text = t
                    break
                except Exception as e:
                    log(f"[WARN] LLM summarize failed {idx} try={attempt+1}: {e}")
                    time.sleep(2 * (attempt + 1))

        if not text:
            # フォールバック台本（URLや記号を除去）
            title = (it.get("title") or "").strip()
            summ = re.sub(r"https?://\S+", "", (it.get("summary") or "").strip())
            summ = re.sub(r"\s+", " ", summ)
            text = (
                f"一人語り：今日は「{title}」。要点はこう。{summ}。"
                "背景では何がボトルネックだったのか、実はデータと運用。"
                "技術の肝は、モデルの前処理と評価設計。実務ではワークフローのどこに刺すかが勝負。"
                "最後に一言、ツールは使い倒して初めて資産。 \n"
                "二人語り：司会『これ、どこがすごい？』 相棒『地味に運用が楽になる点。派手さより現場力！』"
            )

        out.append({"title": it.get("title",""), "text": text})

    log(f"[INFO] texts_for_tts: {len(out)}")
    return out

# ---------- TTS: OpenAI -> espeak-ng fallback ----------

def tts_espeak(text: str, lang: str = "ja", speed_wpm: int = 180, pitch: int = 50) -> AudioSegment:
    """ローカルTTS（ネット不要）。espeak-ng で wav を生成して読み込む。"""
    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "out.wav")
        # espeak-ng -v ja -s 180 -p 50 -w out.wav "text"
        cmd = ["espeak-ng", "-v", lang, "-s", str(speed_wpm), "-p", str(pitch), "-w", wav_path, text]
        subprocess.run(cmd, check=True)
        seg = AudioSegment.from_wav(wav_path)
    return seg

def _openai_tts_request(client, **kwargs):
    """
    SDK差分対策：response_format / format 両対応で呼び分ける。
    """
    try:
        return client.audio.speech.create(**kwargs)
    except TypeError as e:
        msg = str(e)
        if "unexpected keyword argument 'response_format'" in msg:
            kwargs.pop("response_format", None)
            kwargs["format"] = "mp3"
            return client.audio.speech.create(**kwargs)
        if "unexpected keyword argument 'format'" in msg:
            kwargs.pop("format", None)
            kwargs["response_format"] = "mp3"
            return client.audio.speech.create(**kwargs)
        raise

def _bytes_from_openai_tts_response(client, model, voice, text) -> bytes:
    """
    いくつかのSDK版を吸収して mp3 バイトを取り出す。
    """
    # 1) 非ストリーミング（できればこれで）
    try:
        resp = _openai_tts_request(
            client,
            model=model,
            voice=voice,
            input=text,
            response_format="mp3",
        )
        if hasattr(resp, "content") and resp.content:
            return resp.content  # 現行版
        # 古い版のフォールバックいろいろ
        if hasattr(resp, "read"):
            return resp.read()
        if isinstance(resp, (bytes, bytearray)):
            return bytes(resp)
    except Exception as e:
        log(f"[WARN] TTS non-streaming path failed: {e}")

    # 2) ストリーミング
    try:
        from openai import Stream
        tmp_path = os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4().hex}.mp3")
        with client.audio.speech.with_streaming_response.create(
            model=model, voice=voice, input=text, response_format="mp3"
        ) as stream:
            stream.stream_to_file(tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    except Exception as e:
        log(f"[WARN] TTS streaming path failed: {e}")
        raise

def tts_openai_or_fallback(text: str, voice: str, model: str, speaking_rate: float) -> AudioSegment:
    """
    まず OpenAI TTS（分割＆多段互換）。失敗したら espeak-ng にフォールバック。
    """
    use_openai = bool(os.getenv("OPENAI_API_KEY"))
    # チャンク分割
    chunk_size = 700
    lines = text.splitlines()
    chunks, buf = [], ""
    for ln in lines:
        if len(buf) + len(ln) + 1 > chunk_size:
            if buf:
                chunks.append(buf)
                buf = ""
        buf += (ln + " ")
    if buf:
        chunks.append(buf)

    segments: List[AudioSegment] = []

    if use_openai:
        from openai import APIConnectionError
        client = openai_client()
        for i, ch in enumerate(chunks, 1):
            ok = False
            for attempt in range(3):
                try:
                    mp3_bytes = _bytes_from_openai_tts_response(client, model, voice, ch)
                    seg = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
                    segments.append(seg)
                    ok = True
                    break
                except APIConnectionError as e:
                    log(f"[WARN] TTS connection error (retry) part={i}: {e}")
                    time.sleep(2 * (attempt + 1))
                except TypeError as e:
                    log(f"[WARN] TTS signature mismatch part={i}: {e}")
                    time.sleep(1)
                except Exception as e:
                    log(f"[WARN] TTS failed part={i}: {e}")
                    time.sleep(1)
                    break  # このチャンクは諦め、次へ
            if not ok:
                log(f"[WARN] OpenAI TTS fallback to espeak for part={i}")
                try:
                    segments.append(tts_espeak(ch, lang="ja", speed_wpm=int(180 * (1.0 / max(0.5, min(2.0, speaking_rate)))), pitch=50))
                except Exception as e:
                    log(f"[WARN] espeak fallback failed part={i}: {e}")

    else:
        # APIキーなし：全部ローカルTTS
        for ch in chunks:
            segments.append(tts_espeak(ch, lang="ja", speed_wpm=180, pitch=50))

    # 結合
    segments = [s for s in segments if s is not None]
    if not segments:
        raise RuntimeError("TTSがすべて失敗しました。APIキーやネットワーク設定を確認してください。")
    out = segments[0]
    for s in segments[1:]:
        out += s
    return out

# ---------- Build program ----------

def build_program_from_texts(texts: List[Dict[str, Any]], cfg: Dict[str, Any]) -> AudioSegment:
    voice = cfg["tts"]["voice"]
    model = cfg["tts"]["model"]
    rate = float(cfg["tts"]["speaking_rate"])

    intro_text = "通勤AIニュース。今日も技術濃度高めでいきます。"
    outro_text = "以上、通勤AIニュースでした。良い一日を！"

    program = AudioSegment.silent(duration=300)
    try:
        program += tts_openai_or_fallback(intro_text, voice, model, rate)
    except Exception as e:
        log(f"[WARN] intro TTS failed: {e}")

    item_ok = 0
    for i, t in enumerate(texts, 1):
        try:
            program += AudioSegment.silent(duration=200)
            head = f"トピック {i}。{t['title']}。"
            seg = tts_openai_or_fallback(head + "\n" + t["text"], voice, model, rate)
            program += seg
            item_ok += 1
        except Exception as e:
            log(f"[WARN] item TTS failed {i}: {e}")

    try:
        program += AudioSegment.silent(duration=200)
        program += tts_openai_or_fallback(outro_text, voice, model, rate)
    except Exception as e:
        log(f"[WARN] outro TTS failed: {e}")

    if item_ok == 0:
        log("[WARN] no items succeeded; building fallback audio")
        fallback = "今日は技術的な問題で短縮版です。明日はもっと濃い話をお届けします。"
        program = tts_openai_or_fallback(fallback, voice, model, rate)

    return program

# ---------- RSS ----------

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

    ensure_dir(out_dir)
    xml = fg.rss_str(pretty=True)
    with open(os.path.join(out_dir, "feed.xml"), "wb") as f:
        f.write(xml)

# ---------- main ----------

def main():
    validate_openai_api_key()

    cfg = read_yaml("config.yaml")

    base_url = sanitize_base_url(cfg["site"].get("base_url")) or \
               sanitize_base_url(os.getenv("FEED_BASE_URL")) or \
               sanitize_base_url(infer_pages_base_url())
    if not base_url:
        raise RuntimeError("base_url を決定できませんでした。config.yaml か Secrets:FEED_BASE_URL を設定してください。")
    log(f"[INFO] base_url = {base_url}")

    items = fetch_items_from_config(cfg)
    items = filter_items(items, cfg)
    items = pick_for_duration(items, cfg)

    texts = summarize_with_openai(items, cfg)

    out_dir = "output"
    ensure_dir(out_dir)
    today = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d")
    mp3_name = f"{today}-ai-commute.mp3"
    mp3_path = os.path.join(out_dir, mp3_name)

    program = build_program_from_texts(texts, cfg)
    program.export(mp3_path, format="mp3", bitrate=f"{cfg['tts']['output_bitrate']}k")
    log(f"[INFO] mp3 saved: {mp3_path} size={os.path.getsize(mp3_path)}")

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
