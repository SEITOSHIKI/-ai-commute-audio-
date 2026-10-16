"""A4 portrait handout: 8 slides per side (2 columns x 4 rows). Usage: python3 handout8.py slides_dir out.html "タイトル"
slides_dir holds s-01.png ... rendered from the deck's PDF."""
import sys, pathlib, base64
d, out, title = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
imgs = sorted(d.glob("s-*.png"))
uri = lambda p: "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()
pages = []
for k in range(0, len(imgs), 8):
    cells = "".join(f'<figure><img src="{uri(p)}"></figure>' for p in imgs[k:k + 8])
    side = "表" if k == 0 else "裏" if k == 8 else f"{k // 8 + 1}面"
    pages.append(f'<section class="page"><header>{title}　{side}：p.{k + 1}〜{min(k + 8, len(imgs))}</header><div class="grid">{cells}</div></section>')
html = f'''<!doctype html><html><head><meta charset="utf-8"><style>
@page {{ size: A4 portrait; margin: 6mm; }}
body {{ margin:0; font-family: sans-serif; }}
.page {{ page-break-after: always; }}
.page:last-child {{ page-break-after: auto; }}
header {{ font-size: 7.5pt; color:#555; margin-bottom: 1.5mm; }}
.grid {{ display:grid; grid-template-columns: 1fr 1fr; gap: 2.5mm 2.5mm; }}
figure {{ margin:0; border:0.25mm solid #bbb; }}
img {{ width:100%; display:block; }}
</style></head><body>{"".join(pages)}</body></html>'''
pathlib.Path(out).write_text(html)
