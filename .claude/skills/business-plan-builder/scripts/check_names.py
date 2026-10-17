#!/usr/bin/env python3
"""載せない名前（発表者名・社内の名前など）が、資料・台本・ファイル名・Gitのコミットメッセージに残っていないかを調べる。

使い方:
  python3 check_names.py <フォルダ> <名前> [<名前> ...] [--git]
  python3 check_names.py <フォルダ> --names-file names.txt [--git]   # 名前を1行に1つ（コマンド履歴に名前を残したくないとき）

調べるもの: pptx / docx / xlsx（中のXML）、pdf（pdftotext があれば）、md / js / html / txt / json / yaml / csv、ファイル名。
--git を付けると、そのフォルダのGitのコミットメッセージも調べる（過去のコミットに残っていたら、履歴の書き換えが要るかを本人に確かめる）。
見つかったら一覧を出して終了コード1。PDFの文字の間の空白は詰めてから比べる。
"""
import re
import subprocess
import sys
import zipfile
from pathlib import Path

TEXT = {".md", ".js", ".html", ".htm", ".txt", ".json", ".yaml", ".yml", ".csv", ".py", ".xml"}
ZIPPED = {".pptx", ".docx", ".xlsx"}
SKIP_DIRS = {"node_modules", ".git", "__pycache__"}


def squash(s: str) -> str:
    return re.sub(r"\s+", "", s)


def read_zipped(p: Path) -> str:
    out = []
    with zipfile.ZipFile(p) as z:
        for n in z.namelist():
            if n.endswith(".xml") or n.endswith(".rels"):
                xml = z.read(n).decode("utf-8", "ignore")
                out.append(re.sub(r"<[^>]+>", "", xml))
    return "\n".join(out)


def read_pdf(p: Path) -> str:
    try:
        return subprocess.run(["pdftotext", "-q", str(p), "-"], capture_output=True, text=True, timeout=120).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    git = "--git" in args
    args = [a for a in args if a != "--git"]
    root = Path(args[0])
    names = args[1:]
    if names[:1] == ["--names-file"]:
        names = [l.strip() for l in Path(names[1]).read_text(encoding="utf-8").splitlines() if l.strip()]
    names = [n for n in names if n]
    if not names:
        print("名前を1つ以上指定してください")
        return 2
    keys = list({squash(n): (n, squash(n)) for n in names}.values())  # 空白違いは1つにまとめる
    hits = []
    for p in sorted(root.rglob("*")):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        for n, k in keys:
            if k in squash(p.name):
                hits.append((str(p), "ファイル名", n))
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        try:
            if ext in TEXT:
                body = p.read_text(encoding="utf-8", errors="ignore")
            elif ext in ZIPPED:
                body = read_zipped(p)
            elif ext == ".pdf":
                body = read_pdf(p)
            else:
                continue
        except (zipfile.BadZipFile, OSError):
            continue
        flat = squash(body)
        for n, k in keys:
            if k in flat:
                hits.append((str(p), "中身", n))
    if git:
        try:
            log = subprocess.run(["git", "-C", str(root), "log", "--all", "--format=%h%x1f%B%x1e"], capture_output=True, text=True).stdout
            for rec in log.split("\x1e"):
                if "\x1f" not in rec:
                    continue
                h, msg = rec.strip().split("\x1f", 1)
                for n, k in keys:
                    if k in squash(msg):
                        hits.append((h, "コミットメッセージ", n))
        except FileNotFoundError:
            pass
    if hits:
        print(f"見つかりました（{len(hits)}件）")
        for where, kind, n in hits:
            print(f"  [{kind}] {where}  ← 「{n[0]}…」")
        return 1
    print(f"見つかりませんでした（{len(names)}語・{root}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
