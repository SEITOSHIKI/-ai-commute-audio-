# -*- coding: utf-8 -*-
import os
import io
import re
import time
import math
import random
import urllib.parse as up
import datetime as dt
from typing import List, Dict, Tuple

import yaml
import feedparser
from feedgen.feed import FeedGenerator
from pydub import AudioSegment
import trafilatura

# =========================
# 基本ユーティリティ
# =========================
def load_config(path: str = "config.yaml") -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_env_strict(name: str) -> str:
    v = os.getenv(name, "")
    if not v:
        raise RuntimeError(f"{name} が未設定です（Settings → Secrets and variables → Actions に追加）。")
    return v

def sanitize_openai_key(raw: str) -> str:
    key = raw.strip().strip('"').strip("'").replace("\r", "").replace("\n", "")
    if any(ord(c) < 33 or ord(c) == 127 for c in key):
        raise RuntimeError("OPENAI_API_KEY に改行/制御文字/空白が混入しています。貼り直してください。")
    if not (key.startswith("sk-") or key.startswith("sess-")):
        print("[WARN] OPENAI_API_KEY が一般的な形式（sk-/sess-）ではありません。")
    return key

def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

# =========================
# ニュース取得
# =========================
def _gn_rss_url(query: str) -> str:
    cleaned = re.sub(r"\bwhen:\S+\b", "", query).strip()
    return "https://news.google.com/rss/search?q=" + up.quote(cleaned) + "&hl=ja&gl=JP&ceid=JP:ja"

def fetch_items(cfg: Dict) -> List[Dict]:
    items = []
    for q in cfg.get("sources", {}).get("google_news_queries", []):
        d = feedparser.parse(_gn_rss_url(q))
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
    # 重複除去
    seen = set(); deduped = []
    for it in items:
        key = it["link"] or (it["title"], it["published"])
        if key in seen: continue
        seen.add(key); deduped.append(it)
    return deduped

# =========================
# フィルタリング
# =========================
def passes_filters(title: str, summary: str, cfg: Dict) -> bool:
    fcfg = cfg.get("filters", {})
    inc = [w.lower() for w in fcfg.get("must_include_any", [])]
    exc = [w.lower() for w in fcfg.get("must_exclude_any", [])]
    text = f"{title} {summary}".lower()
    if inc and not any(w in text for w in inc): return False
    if any(w in text for w in exc): return False
    return True

def filter_items(items: List[Dict], cfg: Dict) -> List[Dict]:
    return [it for it in items if passes_filters(it["title"], it["summary"], cfg)]

# =========================
# コンテンツ抽出＆整形
# =========================
def fetch_article_text(url: str) -> str:
    try:
        downloaded = trafilatura.fetch_url(url, no_ssl=True, timeout=20)
        extracted = trafilatura.extract(downloaded) if downloaded else ""
        return (extracted or "").strip()
    except Exception:
        return ""

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HEXISH_RE = re.compile(r"\b[0-9A-Za-z]{8,}\b")  # 長い英数字列
CODE_LINE_RE = re.compile(r"^[>\-\*\#\s]*[`~]{1,3}.*$", re.MULTILINE)

def clean_for_speech(text: str) -> str:
    # URL/コード/長い英数字を除去・簡約
    text = URL_RE.sub("", text)
    text = CODE_LINE_RE.sub("", text)
    text = HEXISH_RE.sub("（識別子）", text)
    # 連続空白を整える
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# =========================
# 文字数ターゲット計算（尺→文字）
# =========================
def plan_lengths(n_items: int, cfg: Dict, speaking_rate: float) -> Tuple[int, int, List[int], int]:
    sch = cfg.get("schedule", {})
    tgt_total = int(sch.get("target_total_minutes", 60)) * 60
    s_min = int(sch.get("per_item_seconds_min", 120))
    s_max = int(sch.get("per_item_seconds_max", 360))
    bonus_sec = int(cfg.get("summary", {}).get("bonus_tip_seconds", 0)) if cfg.get("summary", {}).get("include_bonus_tip") else 0

    # 1項目の目安秒数（全体で均等配分しつつ、min/maxでクリップ）
    if n_items > 0:
        ideal = max(s_min, min(s_max, (tgt_total - bonus_sec) // max(1, n_items)))
    else:
        ideal = s_min

    # 日本語の概算：1秒 ≒ 9.5文字（話速1.0相当）。話速に応じて補正
    chars_per_sec = 9.5 / max(0.5, speaking_rate)
    per_item_chars = int(ideal * chars_per_sec)

    # 全項目の配分（端数はランダムで散らす）
    lens = [per_item_chars] * n_items
    drift = int(per_item_chars * 0.2)
    for i in range(n_items):
        lens[i] = max(int(s_min * chars_per_sec), min(int(s_max * chars_per_sec), per_item_chars + random.randint(-drift, drift)))

    bonus_chars = int(bonus_sec * chars_per_sec) if bonus_sec > 0 else 0
    return ideal, per_item_chars, lens, bonus_chars

# =========================
# LLM（要約・生成）
# =========================
def summarize_with_openai(title: str, article: str, style: str, target_chars: int, temperature: float) -> str:
    from openai import OpenAI
    api_key = sanitize_openai_key(get_env_strict("OPENAI_API_KEY"))
    base_url = os.getenv("OPENAI_BASE_URL", None)
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=60, max_retries=1)

    sys = (
        "あなたは上級のAI技術ジャーナリスト兼ナレーターです。"
        "要望に厳密に従い、日本語で一人語りの原稿を作成します。"
        "URL/生アドレス/長い英数字列/コード/逐次スペル読みは出力禁止です。"
        "略語は自然な日本語やカタカナで言い換えてください（例: GPT→ジーピーティー）。"
    )
    user = (
        f"【見出し】{title}\n\n"
        f"【参考本文（必要に応じて利用）】\n{article}\n\n"
        "【執筆方針】\n"
        f"{style}\n\n"
        "【長さの目安】\n"
        f"およそ {max(600, target_chars)} 文字（±20%）。\n"
    )
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4o-mini"),
        temperature=float(temperature),
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
    )
    text = (resp.choices[0].message.content or "").strip()
    return clean_for_speech(text)

def make_bonus_tip(style: str, target_chars: int, temperature: float) -> str:
    from openai import OpenAI
    api_key = sanitize_openai_key(get_env_strict("OPENAI_API_KEY"))
    base_url = os.getenv("OPENAI_BASE_URL", None)
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=60, max_retries=1)

    sys = (
        "あなたは上級のAI技術ジャーナリスト兼ナレーターです。"
        "プロンプトエンジニアリング/オーケストレーション/マルチエージェント/自律型エージェント/"
        "GitHubでのAI活用/深津プロンプト応用などから、今日の学びを1つだけ選んで解説します。"
        "URL/生アドレス/長い英数字列/コード/逐次スペル読みは禁止。"
        "一人語りで、実務に持ち帰れる要点と簡単な手順・落とし穴を入れること。"
    )
    user = (
        "【お題】通勤向けAIラジオの締めに、今日の学びを1本。\n"
        f"【長さ目安】およそ {max(400, target_chars)} 文字（±20%）。\n"
        f"【スタイル】\n{style}\n"
    )
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4o-mini"),
        temperature=float(temperature),
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
    )
    return clean_for_speech((resp.choices[0].message.content or "").strip())

# =========================
# TTS（堅牢）
# =========================
def tts_openai(text: str, voice="alloy", model="gpt-4o-mini-tts", speaking_rate=1.05, retries=3, backoff=2.0) -> AudioSegment:
    from openai import OpenAI
    api_key = sanitize_openai_key(get_env_strict("OPENAI_API_KEY"))
    base_url = os.getenv("OPENAI_BASE_URL", None)
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=60, max_retries=0)

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with client.audio.speech.with_streaming_response.create(model=model, voice=voice, input=text) as resp:
                mp3_bytes = resp.read()
            seg = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
            if speaking_rate and speaking_rate != 1.0:
                seg = seg._spawn(seg.raw_data, overrides={"frame_rate": int(seg.frame_rate * speaking_rate)}).set_frame_rate(seg.frame_rate)
            return seg.set_frame_rate(44100).set_channels(2)
        except Exception as e:
            print(f"[WARN] streaming TTS failed (try {attempt}/{retries}): {e}; fallback...")
            last_err = e
            try:
                resp = client.audio.speech.create(model=model, voice=voice, input=text)
                mp3_bytes = getattr(resp, "content", None) or (resp.read() if hasattr(resp, "read") else None)
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
    raise RuntimeError(f"OpenAI TTS に失敗しました: {last_err}")

def synthesize_texts(texts: List[str], voice: str, model: str, rate: float) -> List[AudioSegment]:
    if not texts:
        raise RuntimeError("音声化するテキストが0件です。")
    segs = []
    for i, t in enumerate(texts, 1):
        print(f"[INFO] TTS {i}/{len(texts)} chars={len(t)}")
        segs.append(tts_openai(t, voice=voice, model=model, speaking_rate=rate))
    return segs

# =========================
# プログラム組立 & 出力
# =========================
def build_program(segs: List[AudioSegment]) -> AudioSegment:
    gap = AudioSegment.silent(duration=800)  # 0.8秒
    program = AudioSegment.silent(duration=600)
    for s in segs:
        program += s + gap
    return program

def export_mp3(program: AudioSegment, out_dir: str, bitrate_k: int) -> str:
    os.makedirs(out_dir, exist_ok=True)
    today = now_utc().date().isoformat()
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
    now = now_utc()
    fg.pubDate(now)

    file_name = os.path.basename(mp3_path)
    day = os.path.basename(os.path.dirname(mp3_path))
    rel_url = f"{day}/{file_name}"
    enclosure_url = f"{base_url}/{rel_url}"
    file_len = os.path.getsize(mp3_path) if os.path.exists(mp3_path) else 0

    fe = fg.add_entry()
    fe.id(enclosure_url)
    fe.title(f"{title} {day}")
    fe.enclosure(enclosure_url, file_len, 'audio/mpeg')
    fe.pubDate(now)
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
    summary_cfg = cfg.get("summary", {})
    speaking_rate = float(tts_cfg.get("speaking_rate", 1.05))

    base_url = (os.getenv("FEED_BASE_URL") or site.get("base_url") or "").strip()
    if not base_url:
        raise RuntimeError("FEED_BASE_URL が未設定です（Settings → Secrets → Actions に設定）。")

    all_items = fetch_items(cfg)
    print(f"[INFO] fetched items: {len(all_items)}")
    picked = filter_items(all_items, cfg)[: int(cfg.get("schedule", {}).get("max_items", 10))]
    print(f"[INFO] picked items: {len(picked)}")

    # 尺に基づく文字数割り当て
    _, _, per_item_chars, bonus_chars = plan_lengths(len(picked), cfg, speaking_rate)

    # 各記事の本文抽出 → LLMで一人語り原稿化
    style = summary_cfg.get("style", "")
    temp = float(summary_cfg.get("temperature", 0.4))
    texts: List[str] = []

    # オープニング
    opening = (
        "おはようございます。通勤AIニュースです。今日は人工知能の最新トピックを、"
        "背景や技術の仕組み、実務に使えるヒントまで、一息で分かるようにお届けします。"
        "コーヒーの香りがする方は一口どうぞ。それでは、いきましょう。"
    )
    texts.append(opening)

    for i, it in enumerate(picked, 1):
        base = fetch_article_text(it["link"]) if it.get("link") else ""
        if not base:
            base = it.get("summary", "")
        base = clean_for_speech(base)

        target_chars = per_item_chars[i - 1] if i - 1 < len(per_item_chars) else 1200
        body = summarize_with_openai(it["title"], base, style, target_chars, temp)
        # 念のため最終クレンジング
        body = clean_for_speech(body)
        # セクション間に軽い導入を追加
        preface = f"— トピック {i}。「{it['title']}」"
        texts.append(preface + "。" + body)

        # 軽いウェイト（API叩きすぎ防止）
        time.sleep(0.4)

    # ボーナス学びコーナー
    if summary_cfg.get("include_bonus_tip"):
        tip = make_bonus_tip(style, bonus_chars or 600, temp)
        tip = "締めに、今日の学び。一つだけ覚えて帰りましょう。" + tip
        texts.append(tip)

    # クロージング
    closing = (
        "以上、通勤AIニュースでした。今日も良い一日を。新しいアイデアが浮かんだら、"
        "メモして昼休みに試してみてください。それでは、行ってらっしゃい。"
    )
    texts.append(closing)

    print(f"[INFO] texts_for_tts (incl. intro/outro/bonus): {len(texts)}")

    # 合成
    segs = synthesize_texts(
        texts,
        voice=tts_cfg.get("voice", "alloy"),
        model=tts_cfg.get("model", "gpt-4o-mini-tts"),
        rate=speaking_rate,
    )
    program = build_program(segs)

    today = now_utc().date().isoformat()
    out_dir = os.path.join("output", today)
    mp3_path = export_mp3(program, out_dir, int(tts_cfg.get("output_bitrate", 128)))

    build_and_save_podcast(mp3_path, out_dir, base_url, site.get("title", "通勤AIニュース"), site.get("author", "あなた"))

if __name__ == "__main__":
    main()
