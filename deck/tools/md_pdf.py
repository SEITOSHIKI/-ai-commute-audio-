"""Turn a simple md (headings, numbered/bullet lists, tables, bold, code) into print HTML. Usage: python3 tools/md_pdf.py in.md out.html
Then: node tools/html_pdf.js out.html out.pdf"""
import html, re, sys
src, dst = sys.argv[1], sys.argv[2]
def inline(t):
    t = html.escape(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
    return t.replace("【要決定】", '<span class="tag">要決定</span>').replace("【要確認】", '<span class="tag">要確認</span>')
out, lst, tbl = [], None, []
def close():
    global lst
    if lst: out.append(f"</{lst}>"); lst = None
def flush_tbl():
    if not tbl: return
    rows = [r for r in tbl if not re.match(r"^\|[-|\s:]+\|$", r)]
    cells = [[c.strip() for c in r.strip("|").split("|")] for r in rows]
    out.append("<table>" + "".join("<tr>" + "".join(f"<{'th' if i == 0 else 'td'}>{inline(c)}</{'th' if i == 0 else 'td'}>" for c in r) + "</tr>" for i, r in enumerate(cells)) + "</table>")
    tbl.clear()
for line in open(src, encoding="utf-8").read().splitlines():
    if line.startswith("|"): close(); tbl.append(line); continue
    flush_tbl()
    if m := re.match(r"^(#{1,3}) (.*)", line): close(); n = len(m[1]); out.append(f"<h{n}>{inline(m[2])}</h{n}>")
    elif m := re.match(r"^(\d+)\. (.*)", line):
        if lst != "ol": close(); out.append(f'<ol start="{m[1]}">'); lst = "ol"
        out.append(f"<li value=\"{m[1]}\">{inline(m[2])}</li>")
    elif m := re.match(r"^[-*] (.*)", line):
        if lst != "ul": close(); out.append("<ul>"); lst = "ul"
        out.append(f"<li>{inline(m[1])}</li>")
    elif line.strip(): close(); out.append(f"<p>{inline(line)}</p>")
    else: close()
close(); flush_tbl()
css = """@page{size:A4;margin:14mm 14mm 16mm}body{font-family:'BIZ UDPGothic','IPAPGothic',sans-serif;font-size:10pt;line-height:1.6;color:#1a2433}
h1{font-size:17pt;color:#17305a;border-bottom:2px solid #17305a;padding-bottom:4px;margin:0 0 8px}
h2{font-size:12pt;color:#fff;background:#17305a;padding:4px 8px;margin:16px 0 6px;break-after:avoid}
ol,ul{padding-left:2.2em;margin:4px 0}li{margin:0 0 5px;break-inside:avoid}b{color:#17305a}
code{font-size:9pt;background:#eef2f7;padding:0 3px}.tag{background:#ffd23f;color:#1a2433;font-size:8.5pt;padding:0 4px;border-radius:2px;font-weight:bold}
table{border-collapse:collapse;width:100%;font-size:9pt;margin:6px 0}th{background:#17305a;color:#fff;text-align:left}th,td{border-bottom:1px solid #c9d2de;padding:4px 6px;vertical-align:top}"""
open(dst, "w", encoding="utf-8").write(f"<!doctype html><html lang=ja><meta charset=utf-8><style>{css}</style><body>{''.join(out)}</body></html>")
