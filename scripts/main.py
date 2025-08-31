# -*- coding: utf-8 -*-
import os
import io
import re
import time
import urllib.parse as up
import datetime as dt
from typing import List, Dict

import yaml
import feedparser
from feedgen.feed import FeedGenerator
from pydub import AudioSegment
import trafilatura

# =========================
# 設定読み込み
# =========================
def load_config(path: str = "config.yaml") -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# =========================
# ニュース取得
# =========================
def _gn_rss_url(query: str) -> str:
    cleaned = re.sub(r"\bwhen:\S+\b", "", query).strip()
    return (
        "https://news.google.com/rss/search?"
        + "q=" + up.quote(cleaned)
        + "&hl=ja&gl=JP&ceid=JP:ja"
    )

def fetch_items(cfg: Dict) -> List[Dict]:
    items = []

    for q in cfg.get("sources", {}).get("google_news_queries", []):
        url = _gn_rss_url(q)
        d = feedparser.parse(url)
        for e in d.entries:
            items.append({
                "title": e.get("title", "").strip(),
                "link": e.get("link", "").strip(),
                "summary": e.get("summary", "").strip(),
                "published": e.get("published", ""),
            })

    for url in cfg.get("sources", {}).get("extra_rss", []):
        d = feedparser.parse(url)
        for e in d.entries:
            items.append({
                "title": e.get("title", "").strip(),
                "link": e.get("link", "").strip(),
                "summary": e.get("summary", "").strip(),
                "published": e.get("published", ""),
            })

    seen = set()
    deduped = []
    for it in items:
        key = it["link"] or (it["title"], it["published"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(it)
    return deduped

# =========================
# フィルタリング
# =========================
def passes_filters(title: str, summary: str, cfg: Dict) -> bool:
    fcfg = cfg.get("filters", {})
    inc = [w.lower() for w in fcfg.get("must_include_any", [])]
    exc = [w.lower() for w in fcfg.get("must_exclude_any", [])]
    text = f"{title} {summary}".lower()
    if inc and not any(w in text for w in inc):
        return False
    if any(w in text for w in exc):
        return False
    return True

def filter_items(items: List[Dict], cfg: Dict) -> List[Dict]:
    return [it for it in items if passes_filters(it["title"], it["summary"], cfg)]

# =========================
# テキスト整形
# =========================
def render_texts(items: List[Dict], cfg: Dict) -> List[str]:
    max_chars = cfg.get("summary", {}).get("max_chars_per_item", 900)
    texts = []
    for i, it in enumerate(items, 1):
        body = it.get("summary") or ""
        try:
            downloaded = trafilatura.fetch_url(it["link"])
            extracted = trafilatura.extract(downloaded) if downloaded else ""
            if extracted:
                body = extracted
        except Exception:
            pass
        blob = f"■ トピック {i}\n要点: {it['title']}\n背景: {body}\nリンク: {it['link']}\n".strip()
        if len(blob) > max_chars:
            blob = blob[:max_chars] + "……"
        texts.append(blob)
    return texts

# =========================
# OpenAI APIキーを安全に取得
# =========================
def get_sanitized_openai_key() -> str:
    raw = os.getenv("OPENAI_API_KEY", "")
    if not raw:
        raise RuntimeError("OPENAI_API_KEY が未設定です（Settings → Secrets and variables → Actions に追加）。")
    # 前後の空白/引用符/改行を除去
    key = raw.strip().strip('"').strip("'").replace("\r", "").replace("\n", "")
    # 制御文字・空白を含んでいないかチェック
    if any(ord(c) < 33 or ord(c) == 127 for c in key):
        raise RuntimeError("OPENAI_API_KEY に改行/制御文字/空白が含まれています。Secrets を貼り直してください（コピペ時の改行/空白/全角に注意）。")
    # 形式は様々（sk- / etc）なので prefixは強制しないがヒントは出す
    if not (key.startswith("sk-") or key.startswith("sess-")):
        print("[WARN] OPENAI_API_KEY が一般的な形式（sk-/sess-）ではありません。値を再確認してください。")
    return key

# =========================
# TTS（堅牢化）
# =========================
def tts_openai(text: str, voice="alloy", model="gpt-4o-mini-tts", speaking_rate=1.05, retries=3, backoff=2.0) -> AudioSegment:
    from openai import OpenAI

    api_key = get_sanitized_openai_key()
    # base_url があれば使用（Azure/OpenAI互換環境など）
    base_url = os.getenv("OPENAI_BASE_URL", None)
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=60, max_retries=0)

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            # まずストリーミング（安定 & メモリ効率）
            with client.audio.speech.with_streaming_response.create(
                model=model, voice=voice, input=text
            ) as resp:
                mp3_bytes = resp.read()
            seg = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
            if speaking_rate and speaking_rate != 1.0:
                seg = seg._spawn(seg.raw_data, overrides={"frame_rate": int(seg.frame_rate * speaking_rate)}).set_frame_rate(seg.frame_rate)
            return seg.set_frame_rate(44100).set_channels(2)
        except Exception as e:
            print(f"[WARN] streaming TTS failed (try {attempt}/{retries}): {e}; fallback to non-streaming...")
            last_err = e
            # 非ストリーミングのフォールバック
            try:
                resp = client.audio.speech.create(model=model, voice=voice, input=text)
                # 新版SDKは content を持つ
                mp3_bytes = getattr(resp, "content", None)
                if not mp3_bytes and hasattr(resp, "read"):
                    mp3_bytes = resp.read()
                if not mp3_bytes:
                    raise RuntimeError("TTSレスポンスから音声データを取得できませんでした。")
                seg = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
                if speaking_rate and speaking_rate != 1.0:
                    seg = seg._spawn(seg.raw_data, overrides={"frame_rate": int(seg.frame_rate * speaking_rate)}).set_frame_rate(seg.frame_rate)
                return seg.set_frame_rate(44100).set_channels(2)
            except Exception as e2:
                last_err = e2
                if attempt < retries:
                    time.sleep(backoff ** attempt)
                else:
                    break
    # ここまで来たら致命的
    raise RuntimeError(f"OpenAI TTS に失敗しました: {last_err}")

def synthesize_items(texts: List[str], voice: str, model: str, rate: float) -> List[AudioSegment]:
    if not texts:
        raise RuntimeError("音声化するテキストが0件です（フィルタが厳しすぎる可能性）。")
    segs = []
    for i, t in enumerate(texts, 1):
        print(f"[INFO] TTS {i}/{len(texts)} chars={len(t)}")
        segs.append(tts_openai(t, voice=voice, model=model, speaking_rate=rate))
    return segs

# =========================
# MP3 / フィード生成
# =========================
def build_program(segs: List[AudioSegment]) -> AudioSegment:
    silence = AudioSegment.silent(duration=600)  # 0.6s
    program = AudioSegment.silent(duration=600)
    for s in segs:
        program += s + silence
    return program

def export_mp3(program: AudioSegment, out_dir: str, bitrate_k: int) -> str:
    os.makedirs(out_dir, exist_ok=True)
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    mp3_name = f"{today}-ai-commute.mp3"
    mp3_path = os.path.join(out_dir, mp3_name)
    program.export(mp3_path, format="mp3", bitrate=f"{bitrate_k}k")
    return mp3_path

def build_and_save_podcast(mp3_path: str, out_dir: str, base_url: str, title: str, author: str):
    base_url = (base_url or "").strip().rstrip("/")

    fg = FeedGenerator(); fg.load_extension('podcast')
    fg.id(f"{base_url}/feed.xml")
    fg.title(title)
    fg.link(href=base_url)
    fg.link(href=f"{base_url}/feed.xml", rel='self')
    fg.description("通勤向け：AI関連ニュースの音声ダイジェスト（日本語）")
    fg.language('ja')
    now_utc = dt.datetime.now(dt.timezone.utc)
    fg.pubDate(now_utc)

    file_name = os.path.basename(mp3_path)
    day = os.path.basename(os.path.dirname(mp3_path))
    rel_url = f"{day}/{file_name}"
    enclosure_url = f"{base_url}/{rel_url}"
    file_len = os.path.getsize(mp3_path) if os.path.exists(mp3_path) else 0

    fe = fg.add_entry()
    fe.id(enclosure_url)
    fe.title(f"{title} {day}")
    fe.enclosure(enclosure_url, file_len, 'audio/mpeg')
    fe.pubDate(now_utc)
    fe.description("本日の主要AIニュースを、日本語でやさしく要点解説。")

    xml = fg.rss_str(pretty=True)
    with open(os.path.join(out_dir, "feed.xml"), "wb") as f:
        f.write(xml)
    root_output = os.path.dirname(out_dir)
    with open(os.path.join(root_output, "feed.xml"), "wb") as f:
        f.write(xml)

# =========================
# メイン
# =========================
def main():
    cfg = load_config()
    site = cfg.get("site", {})
    tts_cfg = cfg.get("tts", {})
    sched = cfg.get("schedule", {})
    max_items = int(sched.get("max_items", 10))

    base_url = (os.getenv("FEED_BASE_URL") or site.get("base_url") or "").strip()
    if not base_url:
        raise RuntimeError("FEED_BASE_URL が未設定です（Settings → Secrets → Actions に設定）。")

    all_items = fetch_items(cfg)
    print(f"[INFO] fetched items: {len(all_items)}")
    picked = filter_items(all_items, cfg)[:max_items]
    print(f"[INFO] picked items: {len(picked)}")

    texts = render_texts(picked, cfg)
    print(f"[INFO] texts_for_tts: {len(texts)}")

    segs = synthesize_items(
        texts,
        voice=tts_cfg.get("voice", "alloy"),
        model=tts_cfg.get("model", "gpt-4o-mini-tts"),
        rate=float(tts_cfg.get("speaking_rate", 1.05)),
    )
    program = build_program(segs)

    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    out_dir = os.path.join("output", today)
    mp3_path = export_mp3(program, out_dir, int(tts_cfg.get("output_bitrate", 128)))

    build_and_save_podcast(mp3_path, out_dir, base_url, site.get("title", "通勤AIニュース"), site.get("author", "あなた"))

if __name__ == "__main__":
    main()
