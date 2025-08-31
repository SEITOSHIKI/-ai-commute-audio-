# -*- coding: utf-8 -*-
import os, io, re, time, random, math, datetime as dt, urllib.parse as up
from typing import List, Dict, Tuple

import yaml, feedparser, trafilatura
from feedgen.feed import FeedGenerator
from pydub import AudioSegment

# --------------------------- 共通ユーティリティ ---------------------------

def load_config(path: str = "config.yaml") -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def get_env_strict(name: str) -> str:
    v = os.getenv(name, "")
    if not v:
        raise RuntimeError(f"{name} が未設定です（Settings → Secrets and variables → Actions に追加）。")
    return v

def sanitize_openai_key(raw: str) -> str:
    key = raw.strip().strip('"').strip("'").replace("\r", "").replace("\n", "")
    if any(ord(c) < 33 or ord(c) == 127 for c in key):
        raise RuntimeError("OPENAI_API_KEY に改行/制御文字/空白が混入。貼り直してください。")
    return key

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HEXISH_RE = re.compile(r"\b[0-9A-Za-z]{8,}\b")
CODE_LINE_RE = re.compile(r"^[>\-\*\#\s]*[`~]{1,3}.*$", re.MULTILINE)

def clean_for_speech(text: str) -> str:
    text = URL_RE.sub("", text)
    text = CODE_LINE_RE.sub("", text)
    text = HEXISH_RE.sub("（識別子）", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# --------------------------- ニュース取得 ---------------------------

def _gn_rss_url(query: str) -> str:
    cleaned = re.sub(r"\bwhen:\S+\b", "", query).strip()
    return "https://news.google.com/rss/search?q=" + up.quote(cleaned) + "&hl=ja&gl=JP&ceid=JP:ja"

def fetch_bucket_items(bucket_name: str, bucket_cfg: Dict) -> List[Dict]:
    items = []
    for q in bucket_cfg.get("google_news_queries", []) or []:
        d = feedparser.parse(_gn_rss_url(q))
        for e in d.entries:
            items.append({
                "bucket": bucket_name,
                "title": e.get("title", "").strip(),
                "link": e.get("link", "").strip(),
                "summary": e.get("summary", "").strip(),
                "published": e.get("published", ""),
            })
    for url in bucket_cfg.get("extra_rss", []) or []:
        d = feedparser.parse(url)
        for e in d.entries:
            items.append({
                "bucket": bucket_name,
                "title": e.get("title", "").strip(),
                "link": e.get("link", "").strip(),
                "summary": e.get("summary", "").strip(),
                "published": e.get("published", ""),
            })
    # 重複除去
    seen = set(); dedup = []
    for it in items:
        key = it["link"] or (it["title"], it["published"])
        if key in seen: continue
        seen.add(key); dedup.append(it)
    return dedup

def fetch_all_items(cfg: Dict) -> List[Dict]:
    buckets = cfg.get("sources", {}).get("topic_buckets", {}) or {}
    all_items = []
    for bname, bcfg in buckets.items():
        all_items += fetch_bucket_items(bname, bcfg)
    print(f"[INFO] fetched total items: {len(all_items)}")
    return all_items

# --------------------------- フィルタリング & 選定 ---------------------------

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
    return [it for it in items if passes_filters(it["title"], it.get("summary",""), cfg) or it["bucket"] not in ("ai_global","ai_jp_art")]

def select_mixed_items(items: List[Dict], cfg: Dict) -> List[Dict]:
    """
    各バケットの required_min を満たすようにピックし、足りなければ他で補完。
    また、トピック不足時は 'synthetic' フラグ付きのプレースホルダを生成して穴埋め。
    """
    buckets_cfg = cfg.get("sources", {}).get("topic_buckets", {}) or {}
    by_bucket: Dict[str,List[Dict]] = {}
    for it in items:
        by_bucket.setdefault(it["bucket"], []).append(it)

    max_items = int(cfg.get("schedule", {}).get("max_items", 5))
    picked: List[Dict] = []

    # 1) required_min を確保
    for bname, bcfg in buckets_cfg.items():
        need = int(bcfg.get("required_min", 0))
        pool = by_bucket.get(bname, [])
        random.shuffle(pool)
        for it in pool[:need]:
            if len(picked) < max_items:
                picked.append(it)

    # 2) 余った枠を人気バケットから（AI系を優先）
    priority = ["ai_global","ai_jp_art","it_fe","career_accounting","accounting_boki3","fitness"]
    for bname in priority:
        if len(picked) >= max_items: break
        pool = by_bucket.get(bname, [])
        random.shuffle(pool)
        for it in pool:
            if len(picked) >= max_items: break
            if it not in picked:
                picked.append(it)

    # 3) 足りなければ合成トピックで埋める（synthetic）
    topics_for_synth = [
        ("accounting_boki3","簿記3級の要点（仕訳・試算表・財務三表のつながり）"),
        ("career_accounting","経理の志望動機・転職市場の今と差別化ポイント"),
        ("it_fe","基本情報技術者（アルゴリズム/ネットワーク/セキュリティ）の頻出論点"),
        ("fitness","筋トレ：週3・45分で伸ばすメニューと栄養の基本"),
        ("ai_global","海外AIトレンドの俯瞰と日本企業への示唆"),
        ("ai_jp_art","松尾研/落合陽一に見る日本発のAIと表現の交差点")
    ]
    t_idx = 0
    while len(picked) < max_items and t_idx < len(topics_for_synth):
        b, title = topics_for_synth[t_idx]
        picked.append({"bucket": b, "title": title, "link": "", "summary": "", "published": "", "synthetic": True})
        t_idx += 1

    # 4) 上限カット
    return picked[:max_items]

# --------------------------- 文字量計画 ---------------------------

def plan_lengths(n_items: int, cfg: Dict, speaking_rate: float) -> Tuple[List[int], int]:
    sch = cfg.get("schedule", {})
    tgt_total = int(sch.get("target_total_minutes", 32)) * 60
    s_min = int(sch.get("per_item_seconds_min", 200))
    s_max = int(sch.get("per_item_seconds_max", 420))
    bonus_sec = int(cfg.get("summary", {}).get("bonus_tip_seconds", 0)) if cfg.get("summary", {}).get("include_bonus_tip") else 0

    if n_items <= 0: return [], 0
    ideal = max(s_min, min(s_max, (tgt_total - bonus_sec) // n_items))
    chars_per_sec = 9.5 / max(0.5, speaking_rate)
    base_chars = int(ideal * chars_per_sec)

    lens = []
    drift = int(base_chars * 0.25)
    for _ in range(n_items):
        lens.append(max(int(s_min*chars_per_sec), min(int(s_max*chars_per_sec), base_chars + random.randint(-drift, drift))))
    bonus_chars = int(bonus_sec * chars_per_sec) if bonus_sec>0 else 0
    return lens, bonus_chars

# --------------------------- OpenAI ラッパ ---------------------------

def openai_client():
    from openai import OpenAI
    api_key = sanitize_openai_key(get_env_strict("OPENAI_API_KEY"))
    base_url = os.getenv("OPENAI_BASE_URL", None)
    return OpenAI(api_key=api_key, base_url=base_url, timeout=60, max_retries=1)

def llm_mono(title: str, base_text: str, style: str, target_chars: int, temperature: float) -> str:
    client = openai_client()
    sys = (
        "あなたは上級のAI技術ジャーナリスト兼ナレーター。"
        "日本語で一人語りの原稿を作る。URL/生アドレス/長い英数字/コード/逐次スペル読みは出力禁止。"
        "海外/国内の視点を織り交ぜ、実務に持ち帰れる学びを入れる。"
    )
    user = (
        f"【見出し】{title}\n"
        f"【参考本文】{base_text}\n"
        f"【スタイル】\n{style}\n"
        f"【長さ目安】約{max(600, target_chars)}文字（±20%）"
    )
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4o-mini"),
        temperature=float(temperature),
        messages=[{"role":"system","content":sys},{"role":"user","content":user}],
    )
    return clean_for_speech((resp.choices[0].message.content or "").strip())

def llm_duo(title: str, base_text: str, style: str, target_chars: int, temperature: float) -> str:
    client = openai_client()
    sys = (
        "あなたは日本語の台本作家。MCとAIエンジニアの2人会話スクリプトを作る。"
        "会話は自然でテンポ良く、笑いどころ少し。URL/生アドレス/長い英数字/コード/逐次スペル読みは禁止。"
        "海外/国内、日本の研究・アート文脈も織り込む。"
    )
    user = (
        f"【見出し】{title}\n"
        f"【参考本文】{base_text}\n"
        f"【スタイル】\n{style}\n"
        "【登場人物】MC（聞き手/要約）／AIエンジニア（技術深掘り）\n"
        "【トーン】砕けた口語、技術濃度高め、ユーモア少々\n"
        f"【長さ目安】約{max(700, target_chars)}文字（±20%）\n"
        "【フォーマット】\n"
        "MC: ……\n"
        "AI: ……\n"
        "（以降交互に数往復、最後は短いオチ）"
    )
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4o-mini"),
        temperature=float(temperature),
        messages=[{"role":"system","content":sys},{"role":"user","content":user}],
    )
    return clean_for_speech((resp.choices[0].message.content or "").strip())

def llm_lesson(bucket: str, style: str, target_chars: int, temperature: float) -> str:
    """ニュース不足時に埋める定番レッスン生成"""
    topics = {
        "accounting_boki3": "簿記3級の基礎（仕訳→試算表→財務三表のつながり）を、例題と暗記コツ付きで。",
        "career_accounting": "経理の志望動機・転職の考え方。自分の強みを具体事例で紐づけるフレーム。",
        "it_fe": "基本情報技術者の頻出論点（アルゴリズム、ネットワーク、セキュリティ）を1分要点で。",
        "fitness": "筋トレ週3・45分のメニュー設計（プッシュ/プル/レッグ）と栄養5原則。",
        "ai_global": "海外AIトレンドの俯瞰（モデル動向、推論最適化、エージェント化）と日本企業への示唆。",
        "ai_jp_art": "松尾研究室や落合陽一の文脈から、研究×表現が与える影響を日常業務に翻訳。"
    }
    client = openai_client()
    sys = (
        "あなたは日本語の教育コンテンツ作家。実務に役立つ“持ち帰り”を必ず入れる。"
        "URL/生アドレス/長い英数字/逐次スペル読みは禁止。"
    )
    user = (
        f"【お題】{topics.get(bucket, '今日のAI実務Tips')}\n"
        f"【スタイル】\n{style}\n"
        f"【長さ目安】約{max(500, target_chars)}文字（±20%）"
    )
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4o-mini"),
        temperature=float(temperature),
        messages=[{"role":"system","content":sys},{"role":"user","content":user}],
    )
    return clean_for_speech((resp.choices[0].message.content or "").strip())

# --------------------------- 記事本文取得 ---------------------------

def fetch_article_text(url: str) -> str:
    if not url: return ""
    try:
        downloaded = trafilatura.fetch_url(url, no_ssl=True, timeout=20)
        extracted = trafilatura.extract(downloaded) if downloaded else ""
        return (extracted or "").strip()
    except Exception:
        return ""

# --------------------------- TTS ---------------------------

def tts_openai(text: str, voice="alloy", model="gpt-4o-mini-tts", speaking_rate=1.05) -> AudioSegment:
    from openai import OpenAI
    api_key = sanitize_openai_key(get_env_strict("OPENAI_API_KEY"))
    base_url = os.getenv("OPENAI_BASE_URL", None)
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=60, max_retries=0)

    # まずストリーミング
    try:
        with client.audio.speech.with_streaming_response.create(model=model, voice=voice, input=text) as resp:
            mp3_bytes = resp.read()
        seg = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
    except Exception:
        # 非ストリーミングにフォールバック
        resp = client.audio.speech.create(model=model, voice=voice, input=text)
        mp3_bytes = getattr(resp, "content", None) or (resp.read() if hasattr(resp, "read") else None)
        if not mp3_bytes:
            raise RuntimeError("TTSレスポンスから音声データを取得できませんでした。")
        seg = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")

    if speaking_rate and speaking_rate != 1.0:
        seg = seg._spawn(seg.raw_data, overrides={"frame_rate": int(seg.frame_rate * speaking_rate)}).set_frame_rate(seg.frame_rate)
    return seg.set_frame_rate(44100).set_channels(2)

def synthesize_texts(texts: List[str], voice: str, model: str, rate: float) -> AudioSegment:
    if not texts: raise RuntimeError("音声化するテキストが0件です。")
    gap = AudioSegment.silent(duration=700)
    program = AudioSegment.silent(duration=600)
    for i, t in enumerate(texts, 1):
        print(f"[INFO] TTS {i}/{len(texts)} chars={len(t)}")
        program += tts_openai(t, voice=voice, model=model, speaking_rate=rate) + gap
        time.sleep(0.2)
    return program

# --------------------------- フィード出力 ---------------------------

def export_mp3(program: AudioSegment, out_dir: str, filename: str, bitrate_k: int) -> str:
    os.makedirs(out_dir, exist_ok=True)
    mp3_path = os.path.join(out_dir, filename)
    program.export(mp3_path, format="mp3", bitrate=f"{bitrate_k}k")
    return mp3_path

def write_feed(base_url: str, out_dir: str, items: List[Dict], title: str, author: str):
    base_url = (base_url or "").strip().rstrip("/")
    fg = FeedGenerator(); fg.load_extension('podcast')
    fg.id(f"{base_url}/feed.xml")
    fg.title(title)
    fg.link(href=base_url)
    fg.link(href=f"{base_url}/feed.xml", rel='self')
    fg.description("通勤向け：AI関連ニュースの音声ダイジェスト（日本語）")
    fg.language('ja')
    now = now_utc(); fg.pubDate(now)

    for it in items:
        fe = fg.add_entry()
        fe.id(it["url"])
        fe.title(it["title"])
        fe.enclosure(it["url"], it["length"], 'audio/mpeg')
        fe.pubDate(now)
        fe.description(it.get("desc",""))

    xml = fg.rss_str(pretty=True)
    with open(os.path.join(out_dir, "feed.xml"), "wb") as f:
        f.write(xml)

# --------------------------- メイン ---------------------------

def main():
    cfg = load_config()
    site = cfg.get("site", {})
    tts_cfg = cfg.get("tts", {})
    styles = cfg.get("styles", {})
    summary_cfg = cfg.get("summary", {})
    modes = cfg.get("modes", ["mono","duo"])

    base_url = (os.getenv("FEED_BASE_URL") or site.get("base_url") or "").strip()
    if not base_url:
        raise RuntimeError("FEED_BASE_URL が未設定です（Secrets）。例: https://seitoshiki.github.io/-ai-commute-audio-")

    speaking_rate = float(tts_cfg.get("speaking_rate", 1.05))
    all_items = fetch_all_items(cfg)
    filtered = filter_items(all_items, cfg)
    picked = select_mixed_items(filtered, cfg)
    print(f"[INFO] picked items: {len(picked)}")

    # 文字数割り当て
    per_item_chars, bonus_chars = plan_lengths(len(picked), cfg, speaking_rate)
    temp = float(summary_cfg.get("temperature", 0.45))

    # ニュース本文取得
    articles: List[str] = []
    for it in picked:
        txt = fetch_article_text(it.get("link","")) if not it.get("synthetic") else ""
        if not txt: txt = it.get("summary","")
        articles.append(clean_for_speech(txt))

    # モードごとに台本生成
    texts_by_mode: Dict[str, List[str]] = {}

    # オープニング共通
    opening_mono = "おはようございます。通勤AIニュース、一人語りバージョンです。今日も技術濃度高めで、でも肩の力は抜いて行きましょう。"
    opening_duo  = "MC: おはようございます、通勤AIニュース対話版！\nAI: 眠気覚ましに推論最適化の小ネタ、持ってきました。"

    closing_mono = "以上、通勤AIニュース（モノローグ）でした。昼休みに1トピック復習、夕方に1アイデア実践、これで記憶が定着します。いってらっしゃい！"
    closing_duo  = "MC: 以上、対話版でした！\nAI: 今日の一歩がモデルの汎化性能を上げます。それでは、行ってらっしゃい。"

    # 一人語り
    if "mono" in modes:
        texts = [opening_mono]
        for i, it in enumerate(picked):
            base_text = articles[i]
            target_chars = per_item_chars[i] if i < len(per_item_chars) else 900
            if it.get("synthetic"):
                body = llm_lesson(it["bucket"], styles.get("mono",""), target_chars, temp)
            else:
                body = llm_mono(it["title"], base_text, styles.get("mono",""), target_chars, temp)
            preface = f"— トピック {i+1}：「{it['title']}」"
            texts.append(preface + "。" + body)
            time.sleep(0.3)

        if summary_cfg.get("include_bonus_tip"):
            bonus = llm_lesson("ai_global", styles.get("mono",""), bonus_chars or 600, temp)
            texts.append("締めの“今日の学び”。" + bonus)

        texts.append(closing_mono)
        texts_by_mode["mono"] = texts

    # 二人語り
    if "duo" in modes:
        texts = [opening_duo]
        for i, it in enumerate(picked):
            base_text = articles[i]
            target_chars = per_item_chars[i] if i < len(per_item_chars) else 1000
            if it.get("synthetic"):
                body = llm_lesson(it["bucket"], styles.get("duo",""), target_chars, temp)
                # レッスンは会話風に軽く前置きを付ける
                body = f"MC: 次のテーマ「{it['title']}」。\nAI: 了解、要点を会話で手早くいきます。\n" + body
            else:
                body = llm_duo(it["title"], base_text, styles.get("duo",""), target_chars, temp)
            texts.append(body)
            time.sleep(0.3)

        if summary_cfg.get("include_bonus_tip"):
            bonus = llm_lesson("ai_jp_art", styles.get("duo",""), bonus_chars or 600, temp)
            texts.append("MC: 最後に“今日の学び”まとめ。\nAI: " + bonus)

        texts.append(closing_duo)
        texts_by_mode["duo"] = texts

    # 合成→MP3
    today = now_utc().date().isoformat()
    out_dir = os.path.join("output", today)
    os.makedirs(out_dir, exist_ok=True)

    mp3_items_for_feed = []
    for mode, texts in texts_by_mode.items():
        program = synthesize_texts(
            texts,
            voice=tts_cfg.get("voice","alloy"),
            model=tts_cfg.get("model","gpt-4o-mini-tts"),
            rate=speaking_rate,
        )
        fname = f"{today}-ai-commute-{mode}.mp3"
        path = export_mp3(program, out_dir, fname, int(tts_cfg.get("output_bitrate",128)))
        rel_url = f"{today}/{fname}"
        enclosure_url = f"{base_url.rstrip('/')}/{rel_url}"
        length = os.path.getsize(path) if os.path.exists(path) else 0
        title = f"{site.get('title','通勤AIニュース')} {today} [{ '一人語り' if mode=='mono' else '二人語り' }]"
        desc = "本日のダイジェスト（海外/国内/アート×AI＋実務の学び）。"
        mp3_items_for_feed.append({"url": enclosure_url, "title": title, "length": length, "desc": desc})

    # RSS 書き出し（同日に2アイテム）
    write_feed(base_url, out_dir, mp3_items_for_feed, site.get("title","通勤AIニュース"), site.get("author","あなた"))

if __name__ == "__main__":
    main()
