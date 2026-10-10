# Logos

地表線（横棒＋両端の短い杭）の下に、ケーブル断面を表す同心円3重（外・中は線、中心は塗り）。録画ボタンにも読める形。黒一色で成立する。

| ファイル | 用途 | インク |
|---|---|---|
| `georec-mark-black.svg` | 全ページ右上のマーク（幅 `corner-mark`）、白地上の小さな表示 | `ink` #1A1A1A |
| `georec-mark-white.svg` | `ink` の黒ベタ上（表紙左1/3、理念の黒帯） | `paper` #FFFFFF |
| `georec-wordmark-black.svg` | 白地上のフルロゴ（マーク＋GEOREC＋UNDERGROUND ASSET RECORDS） | `ink` |
| `georec-wordmark-white.svg` | 黒ベタ上のフルロゴ | `paper` |

- 幾何は SKILL.md §6 の仕様値（地表線 x4 y30 w112 h6、杭 w5 h8、円 cx60 cy76 r33/20/9、線幅 6/5）。杭は地表線の両端から下向きに置いた。
- ワードマークの文字はテキスト要素（Arial / Helvetica）。PowerPoint に貼る場合は PNG に書き出して使う。
- 4-up 印刷でつぶれないよう、マークは幅 24px（0.25in）未満にしない。

## GEN³ Works（三現ワークス）

社名変更に伴うロゴ。同心円3重（外・中は線、中心は塗り）を、三現（現場・現物・現実）として地表線なしで使う。

| ファイル | 用途 |
|---|---|
| `gen3-mark-black.svg` / `gen3-mark-white.svg` | 全ページ右上のマーク／黒ベタ上のマーク |
| `gen3-wordmark-black.svg` / `gen3-wordmark-white.svg` | マーク＋「GEN³ WORKS」＋「GENBA · GENBUTSU · GENJITSU」 |

## GEN³ Works ロゴ（2026年10月改訂：AR スキャン・キューブ）

現場を、スマートグラス（AR）であらゆる方向から記録し、勘どころをデータ資産として次の担い手へ渡す、という事業を1つの形にした。

- **四隅のカギ括弧**：AR／スマートグラスのファインダー。現場をあらゆる方向から捉える
- **立方体の三つの面**：三現（現場・現物・現実）。側面の黒と灰は物理的な現場
- **銅色の上面（点の格子）**：データになった現場の勘。銅色は電線の導体の色
- **開いた右上の角から流れ出る点**：データ資産が次の担い手（とAI）へ渡っていく

| ファイル | 用途 |
|---|---|
| `gen3-cube-color.svg` / `-mono` / `-reverse` | マーク単体（ファビコン、アプリ、印刷の1色版、黒地） |
| `gen3-logotype-color.svg` / `-mono` / `-reverse` | マーク＋「GEN³ WORKS」＋「三現ワークス」。スライド右下は color |
| `gen3-logotype-tagline.svg` / `-tagline-reverse` | ＋「現場の勘を、データ資産として次の担い手へ。」。表紙・名刺・展示会 |

- 生成は `node build_gen3_logo.js`。幾何（立方体 中心58,66・一辺31、括弧 線幅7、上面の点 4×4・半径2.7）はこのスクリプトが正
- 32px でも括弧と立方体が読める。これより小さい場合は括弧を省いた立方体だけにする
- 旧 `gen3-mark-*` / `gen3-wordmark-*`（同心円）は旧版。新しい資料では使わない
