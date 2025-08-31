import os, io, glob, math, datetime as dt
from dateutil import tz
import yaml
import feedparser, trafilatura
from pydub import AudioSegment
from feedgen.feed import FeedGenerator

JST = tz.gettz("Asia/Tokyo")

# ===== Utils =====
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def today_slug():
    return dt.datetime.now(JST).strftime("%Y-%m-%d")

def build_gnews_url(query: str, lang="ja", country="JP"):
    import urllib.parse as up
    q = up.quote(query)
    return f"https://news.google.com/rss/search?q={q}&hl={lang}&gl={country}&ceid={country}:{lang}"

# ===== Fetch =====
def fetch_all(config: dict):
    items = []
    # Googleニュース検索
    for q in config["sources"].get("google_news_queries", []):
        url = build_gnews_url(q, lang=config["site"].get("language","ja"))
        feed = feedparser.parse(url)
        for e in feed.entries:
            items.append({
                "title": e.get("title",""),
                "link": e.get("link"),
                "published": e.get("published",""),
                "source": "google_news",
                "query": q,
            })
    # 任意のRSS
    for url in config["sources"].get("extra_rss", []):
        feed = feedparser.parse(url)
        for e in feed.entries:
            items.append({
                "title": e.get("title",""),
                "link": e.get("link"),
                "published": e.get("published",""),
                "source": url,
                "query": None,
            })
    # 重複排除
    seen, out = set(), []
    for it in items:
        k = it.get("link")
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out

# ===== Extract =====
def extract_main_text(url: str):
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return ""
        text = trafilatura.extract(downloaded, include_links=False, include_comments=False, favor_recall=True)
        return text or ""
    except Exception:
        return ""

# ===== Summarize (OpenAI) =====
def summarize_with_openai(text: str, title: str, config: dict) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    max_chars = config["summary"]["max_chars_per_item"]
    sys = (
        "あなたは日本語のジャーナリスト。聴きやすい口語で、要点→背景→影響→一言コメントの順で、"
        f"{max_chars}文字以内にまとめます。固有名詞は日本語と原語を併記し、数字は具体的に。"
    )

    prompt = f"""タイトル: {title}

本文（テキスト）:
{text[:120000]}

出力要件:
- 日本語の口語
- 箇条書き2〜4点＋短いまとめ
- 不確かな点は明示
- 出典は最後に1行で
"""

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[{"role":"system","content":sys}, {"role":"user","content":prompt}],
        max_output_tokens=600,
        temperature=float(config["summary"].get("temperature", 0.3)),
    )
    return resp.output_text.strip()

# ===== TTS (OpenAI) =====
def tts_openai(text: str, voice="alloy", model="gpt-4o-mini-tts", speaking_rate=1.05) -> AudioSegment:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    speech = client.audio.speech.create(model=model, voice=voice, input=text)
    mp3_bytes = speech.audio  # プロバイダ仕様に応じて調整
    seg = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
    if speaking_rate and speaking_rate != 1.0:
        seg = seg._spawn(seg.raw_data, overrides={"frame_rate": int(seg.frame_rate * speaking_rate)}).set_frame_rate(seg.frame_rate)
    return seg.set_frame_rate(44100).set_channels(2)

def synthesize_items(texts, voice, model, rate):
    segs = []
    for t in texts:
        segs.append(tts_openai(t, voice=voice, model=model, speaking_rate=rate))
    return segs

def concat_with_bump(segments, bump_ms=400):
    if not segments:
        return AudioSegment.silent(duration=1000)
    out = AudioSegment.silent(duration=300)
    sep = AudioSegment.silent(duration=bump_ms)
    for s in segments:
        out += s + sep
    return out

# ===== Podcast Feed =====
def build_and_save_podcast(mp3_path: str, out_dir: str, base_url: str, title: str, author: str):
    os.makedirs(out_dir, exist_ok=True)
    fg = FeedGenerator(); fg.load_extension('podcast')

    # 必須フィールド + 自身へのatomリンク
    fg.id(f"{base_url}/feed.xml")
    fg.title(title)
    fg.link(href=base_url, rel='alternate')
    fg.link(href=f"{base_url}/feed.xml", rel='self')
    fg.description("通勤向け：AI関連ニュースの音声ダイジェスト（日本語）")
    fg.language('ja')
    now_utc = dt.datetime.now(dt.timezone.utc)
    fg.pubDate(now_utc)

    # エピソード
    file_name = os.path.basename(mp3_path)
    enclosure_url = f"{base_url}/{file_name}"
    file_len = os.path.getsize(mp3_path) if os.path.exists(mp3_path) else 0

    fe = fg.add_entry()
    fe.id(enclosure_url)
    fe.title(f"{title} {dt.datetime.now().strftime('%Y-%m-%d')}")
    fe.enclosure(enclosure_url, file_len, 'audio/mpeg')
    fe.pubDate(now_utc)
    fe.description("本日の主要AIニュースを、日本語でやさしく要点解説。")

    xml = fg.rss_str(pretty=True)
    with open(os.path.join(out_dir, "feed.xml"), "wb") as f:
        f.write(xml)
    with open(os.path.join(os.path.dirname(out_dir), "feed.xml"), "wb") as f:
        f.write(xml)

# ===== Select & Orchestrate =====
def pick_items(items, config):
    must_any = set(config["filters"].get("must_include_any", []))
    no_any = set(config["filters"].get("must_exclude_any", []))

    def ok(it):
        t = (it.get("title") or "").lower()
        if must_any and not any(k.lower() in t for k in must_any):
            return False
        if any(k.lower() in t for k in no_any):
            return False
        return True

    filtered = [it for it in items if ok(it)]
    return filtered[: config["schedule"]["max_items"]]

def main():
    with open(os.path.join(os.path.dirname(__file__), "..", "config.yaml"), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    out_root = os.path.join(os.path.dirname(__file__), "..", "output")
    day = today_slug(); out_dir = os.path.join(out_root, day); ensure_dir(out_dir)

    items = fetch_all(config)
    picked = pick_items(items, config)

    texts_for_tts = []
    for it in picked:
        body = extract_main_text(it["link"])
        if not body:
            continue
        summ = summarize_with_openai(body, it["title"], config)
        script = f"見出し: {it['title'].strip()}\n\n{summ}\n\n出典URL: {it.get('link','')}" if config["summary"].get("include_source", True) else f"見出し: {it['title'].strip()}\n\n{summ}"
        texts_for_tts.append(script)

    voice = config["tts"]["voice"]; model = config["tts"]["model"]; rate = float(config["tts"]["speaking_rate"])
    segs = synthesize_items(texts_for_tts, voice=voice, model=model, rate=rate)

    program = concat_with_bump(segs, bump_ms=400)
    mp3_path = os.path.join(out_dir, f"{day}-ai-commute.mp3")
    program.export(mp3_path, format="mp3", bitrate=f"{config['tts']['output_bitrate']}k")

    base_url = os.getenv("FEED_BASE_URL") or config["site"].get("base_url")
    if not base_url:
        repo = os.getenv("GITHUB_REPOSITORY", "")
        if repo and "/" in repo:
            user, name = repo.split("/", 1)
            base_url = f"https://{user}.github.io/{name}"
        else:
            base_url = "https://example.com/ai-commute-audio"

    build_and_save_podcast(mp3_path, out_dir, base_url, config["site"]["title"], config["site"]["author"])

    # 古い日の削除
    keep_days = int(config["output"].get("keep_days", 14))
    cutoff = dt.datetime.now(JST) - dt.timedelta(days=keep_days)
    for d in sorted([p for p in glob.glob(os.path.join(out_root, "*")) if os.path.isdir(p)]):
        try:
            when = dt.datetime.strptime(os.path.basename(d), "%Y-%m-%d")
            if when < cutoff.replace(tzinfo=None):
                import shutil; shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass

if __name__ == "__main__":
    main()
