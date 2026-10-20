#!/usr/bin/env bash
# 資料一式をつくり直す：pptx → pdf → 台本つきpdf・配布用pdf・台本md → zip
# 使い方：SOFFICE="python3 <pptxスキル>/scripts/office/soffice.py" NODE_PATH=<node_modules> bash tools/build_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SOFFICE=${SOFFICE:-soffice}
TMP=$(mktemp -d)
for b in build_tora.js build_tora16.js build_tora_detail.js build_tora_easy.js build_service.js build_pivot.js build_deck.js; do node "$b" >/dev/null; done
for f in カンツギ_事業計画書_本編 カンツギ_事業計画書_16枚版 カンツギ_事業計画書_計算詳細版 カンツギ_事業計画書_やさしい版 カンツギ_事業計画書_10分16枚版 カンツギ_サービス説明資料 セルフマネジメント課題_事業構想プレゼン; do
  $SOFFICE --headless --convert-to pdf --outdir . "$f.pptx" >/dev/null 2>&1
done
# 台本つき（スライドの下に台本）
for f in カンツギ_事業計画書_本編 カンツギ_事業計画書_16枚版 カンツギ_事業計画書_計算詳細版 カンツギ_事業計画書_やさしい版 カンツギ_事業計画書_10分16枚版 カンツギ_サービス説明資料; do
  d="$TMP/$f"; mkdir -p "$d"; pdftoppm -png -r 110 "$f.pdf" "$d/s"
  label=${f#カンツギ_事業計画書_}; label=${label#カンツギ_}
  python3 tools/notes_html.py "$f.pptx" "$d" "$TMP/$f.html" "$label"
  node tools/html_pdf.js "$TMP/$f.html" "${f}_台本つき.pdf"
done
# 配布用（A4横・1ページ4枚。両面は短辺とじでA4用紙2枚）
for spec in "カンツギ_事業計画書_16枚版:4" "カンツギ_事業計画書_10分16枚版:4"; do
  f=${spec%%:*}; n=${spec##*:}; d="$TMP/h_$f"; mkdir -p "$d"; pdftoppm -png -r 150 "$f.pdf" "$d/s"
  python3 tools/handout.py "$d" "$TMP/h_$f.html" "株式会社カンツギ 事業計画書" "$n"
  node tools/html_pdf.js "$TMP/h_$f.html" "${f}_A4両面配布用.pdf"
done
python3 tools/notes_md.py カンツギ_事業計画書_16枚版.pptx 16枚版_台本.md
python3 tools/notes_md.py カンツギ_事業計画書_10分16枚版.pptx 10分16枚版_台本.md
python3 tools/notes_md.py カンツギ_サービス説明資料.pptx サービス説明資料_台本.md
rm -f カンツギ_資料一式.zip
zip -q -j "$TMP/all.zip" カンツギ_事業計画書_*.pptx カンツギ_事業計画書_*.pdf カンツギ_サービス説明資料*.pptx カンツギ_サービス説明資料*.pdf $(ls カンツギ_計算詳細版_裏取り.* 2>/dev/null) 作業計画書_10月13日-22日.pdf *.md
mv "$TMP/all.zip" カンツギ_資料一式.zip
rm -rf "$TMP"
echo done
