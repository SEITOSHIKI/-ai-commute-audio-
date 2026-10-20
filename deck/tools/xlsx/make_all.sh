#!/usr/bin/env bash
# 裏取りExcel（とPDF）を作り直す：作成 → 再計算 → tora_v2.js と照合 → PDF。出力は deck/ に置く
# 使い方：SK=<xlsx と pptx のスキルがあるフォルダ> bash tools/xlsx/make_all.sh
# inputs.json は tora_v2.js の前提を写したもの。前提を変えたら直す（照合で不一致が出る）
set -euo pipefail
X=$(cd "$(dirname "$0")" && pwd); DECK=$(cd "$X/../.." && pwd)
export SK=${SK:?SK にスキルのフォルダを指定}
W=$(mktemp -d); mkdir -p "$W/scen"; cp "$X/inputs.json" "$W/"
node "$X/dump.js" "$DECK" > "$W/node_values.json"
python3 "$X/build_xlsx.py" "$W" "$W/test.xlsx"
cp "$W/test.xlsx" "$W/recalc.xlsx"
(cd "$SK/xlsx/scripts" && timeout 200 python3 recalc.py "$W/recalc.xlsx" 120)
python3 "$X/verify.py" "$W" "$W/recalc.xlsx"
python3 "$X/scen_test.py" "$W"
# Excelで開くと計算し直すが、値も入れておく（LibreOfficeで保存し直すとフォント名が変わるため）
python3 "$X/inject.py" "$W/test.xlsx" "$W/recalc.xlsx" "$DECK/カンツギ_計算詳細版_裏取り.xlsx"
python3 "$X/verify.py" "$W" "$DECK/カンツギ_計算詳細版_裏取り.xlsx"
# PDFはフォント名を IPA Pゴシック にしたコピーから書き出す（BIZ UDPゴシックが無い環境向け）
mkdir -p "$W/pdf"; python3 "$X/pdf_copy.py" "$DECK/カンツギ_計算詳細版_裏取り.xlsx" "$W/pdf/カンツギ_計算詳細版_裏取り.xlsx"
(cd "$W" && timeout 300 python3 "$SK/pptx/scripts/office/soffice.py" --headless --convert-to pdf --outdir "$W/pdf" "$W/pdf/カンツギ_計算詳細版_裏取り.xlsx" >/dev/null 2>&1)
cp "$W/pdf/カンツギ_計算詳細版_裏取り.pdf" "$DECK/"
rm -rf "$W"; echo done
