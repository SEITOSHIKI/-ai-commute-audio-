# PDF 用のコピー：BIZ UDPゴシック が無い環境なので、書式とヘッダーのフォント名だけ IPAPGothic に置き換える（セルの文字・式・値は同じ）
import sys, zipfile
src, dst = sys.argv[1], sys.argv[2]
n = 0
with zipfile.ZipFile(src) as zi, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zo:
    for item in zi.infolist():
        data = zi.read(item.filename)
        if item.filename == "xl/styles.xml":
            t = data.decode("utf-8"); n += t.count('val="BIZ UDPゴシック"')
            data = t.replace('val="BIZ UDPゴシック"', 'val="IPAPGothic"').encode("utf-8")
        elif item.filename.startswith("xl/worksheets/") and item.filename.endswith(".xml"):
            t = data.decode("utf-8"); n += t.count('"BIZ UDPゴシック,')
            data = t.replace('"BIZ UDPゴシック,', '"IPAPGothic,').encode("utf-8")
        zo.writestr(item, data)
print("replaced", n)
