def build_and_save_podcast(mp3_path: str, out_dir: str, base_url: str, title: str, author: str):
    os.makedirs(out_dir, exist_ok=True)
    fg = FeedGenerator(); fg.load_extension('podcast')
    fg.id(base_url + "/feed.xml"); fg.title(title); fg.author({'name': author})
    fg.link(href=base_url, rel='alternate'); fg.language('ja'); fg.pubDate(dt.datetime.now(dt.timezone.utc))

    fe = fg.add_entry()
    fe.id(base_url + "/" + os.path.basename(mp3_path))
    fe.title(title + " " + dt.datetime.now().strftime("%Y-%m-%d"))
    fe.enclosure(base_url + "/" + os.path.basename(mp3_path), 0, 'audio/mpeg')
    fe.pubDate(dt.datetime.now(dt.timezone.utc))

    xml = fg.rss_str(pretty=True)
    with open(os.path.join(out_dir, "feed.xml"), "wb") as f:
        f.write(xml)
    with open(os.path.join(os.path.dirname(out_dir), "feed.xml"), "wb") as f:
        f.write(xml)

