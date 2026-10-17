"""A4 handout PDF source (HTML). 4 slides per page = A4 landscape 2x2, 8 slides per page = A4 portrait 2x4.
Usage: python3 handout.py slides_dir out.html "タイトル" [4|8]
slides_dir holds s-01.png ... rendered from the deck's PDF. Pages are labeled for duplex printing (1枚目 表／裏, 2枚目 表／裏 ...)."""
import sys, pathlib, base64
d, out, title = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
per = int(sys.argv[4]) if len(sys.argv) > 4 else 4
imgs = sorted(d.glob("s-*.png"))
uri = lambda p: "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()
landscape = per == 4
pages = []
for pi, k in enumerate(range(0, len(imgs), per)):
    cells = "".join(f'<figure><img src="{uri(p)}"><figcaption>{k + j + 1}</figcaption></figure>' for j, p in enumerate(imgs[k:k + per]))
    side = f"{pi // 2 + 1}枚目 {'表' if pi % 2 == 0 else '裏'}"
    pages.append(f'<section class="page"><header><span>{title}</span><span>{side}（p.{k + 1}〜{min(k + per, len(imgs))}）</span></header><div class="grid">{cells}</div></section>')
size = "A4 landscape" if landscape else "A4 portrait"
gap = "4mm 5mm" if landscape else "2.5mm 2.5mm"
html = f'''<!doctype html><html><head><meta charset="utf-8"><style>
@page {{ size: {size}; margin: 7mm; }}
body {{ margin:0; font-family: sans-serif; }}
.page {{ page-break-after: always; }}
.page:last-child {{ page-break-after: auto; }}
header {{ display:flex; justify-content:space-between; font-size: 8pt; color:#555; margin-bottom: 2mm; }}
.grid {{ display:grid; grid-template-columns: 1fr 1fr; gap: {gap}; }}
figure {{ margin:0; border:0.25mm solid #bbb; position:relative; }}
img {{ width:100%; display:block; }}
figcaption {{ display:none; }}
</style></head><body>{"".join(pages)}</body></html>'''
pathlib.Path(out).write_text(html)
