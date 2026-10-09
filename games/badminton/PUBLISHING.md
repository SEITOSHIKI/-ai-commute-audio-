# 家族への共有と Google Play 公開の手順

## 全体像

| 経路 | 対象 | 準備 | 公開までの時間 |
|---|---|---|---|
| A. Web 版（GitHub Pages） | iPhone・Android・PC すべて | このブランチを main にマージするだけ | 数分 |
| B. APK を直接インストール | Android の家族 | タグを1つ打つ | 数分 |
| C. Google Play（内部テスト） | Android の家族（最大100人） | デベロッパー登録＋署名鍵 | 登録審査後すぐ |
| D. Google Play（一般公開） | 誰でも | C＋クローズドテスト 12人×14日 | 最短でも約3週間 |

家族にはまず **A**（全員）と **C**（Android の人）を使い、C のテスターをそのまま D の「12人×14日」に数えるのが最短です。

---

## A. Web 版を家族に共有する

1. このブランチを `main` にマージする。
2. `Rally IQ Android` ワークフローの `web` ジョブが、gh-pages ブランチの `/rally-iq/` に Web 版を置く（音声フィードのファイルはそのまま）。毎朝の `build.yml` も同じ場所に作り直すので消えない。
3. 家族に送る URL:
   - ゲーム: `https://seitoshiki.github.io/-ai-commute-audio-/rally-iq/`
   - プライバシーポリシー: `https://seitoshiki.github.io/-ai-commute-audio-/rally-iq/privacy.html`
4. ホーム画面に追加すると、アプリのように全画面で起動し、2回目以降はオフラインでも動く。
   - iPhone: Safari で開く → 共有ボタン → 「ホーム画面に追加」
   - Android: Chrome で開く → メニュー → 「アプリをインストール」または「ホーム画面に追加」

> リポジトリの Settings → Pages の Source が `gh-pages` ブランチになっていることを確認してください（音声フィードで既に使っていればそのままで OK）。

Claude の Artifact で公開しているページも、そのページの「Share」から家族に共有できます（Claude のアカウント設定によっては、リンクを開く側にもログインが必要です）。

## B. Android の家族に APK を直接渡す

1. `rally-iq-v1.0.0` のようなタグを push する（`git tag rally-iq-v1.0.0 && git push origin rally-iq-v1.0.0`）。
2. ワークフローが GitHub Release を作り、APK を添付する。
3. 家族は Android でその Release ページを開き、APK をダウンロードしてインストール（初回は「この提供元のアプリを許可」をオンにする）。

署名鍵（下の C-2）を設定する前のタグでは debug 署名の APK になります。あとで Play 版に切り替えるときは、debug 版を一度アンインストールしてください（署名が違うため上書きできません）。

## C. Google Play に登録して家族をテスターにする

### C-1. デベロッパーアカウントを作る

- https://play.google.com/console で登録（登録料 25 米ドル・1回のみ、本人確認あり）。
- **個人アカウント**で 2023年11月13日以降に作った場合、一般公開の前に「12人以上のテスターで14日間連続のクローズドテスト」が必須です。組織アカウント（D-U-N-S 番号が必要）はこの要件の対象外です。

### C-2. アップロード鍵を作って GitHub に登録する

鍵は一度なくすと再発行の手続きが必要になるので、パスワード管理ツールなどに必ずバックアップしてください。Docker で作る例:

```bash
docker run --rm -it -v "$PWD:/w" -w /w eclipse-temurin:21 \
  keytool -genkeypair -v -keystore upload.jks -alias upload \
  -keyalg RSA -keysize 2048 -validity 10000

base64 -w0 upload.jks > upload.jks.b64   # macOS は base64 -i upload.jks -o upload.jks.b64
```

GitHub のリポジトリ → Settings → Secrets and variables → Actions に、次の4つを登録します。

| Secret 名 | 値 |
|---|---|
| `RALLYIQ_KEYSTORE_BASE64` | `upload.jks.b64` の中身 |
| `RALLYIQ_KEYSTORE_PASSWORD` | キーストアのパスワード |
| `RALLYIQ_KEY_ALIAS` | `upload` |
| `RALLYIQ_KEY_PASSWORD` | 鍵のパスワード |

`upload.jks` と `.b64` はリポジトリに入れないでください（`.gitignore` 済み）。

### C-3. AAB を作る

Actions → `Rally IQ Android` → Run workflow。終わったら成果物 `rally-iq-android-<番号>` から `rally-iq-1.0.0.aab` をダウンロードします。`versionCode` には実行番号が自動で入ります。

### C-4. Play Console でアプリを作る

1. 「アプリを作成」→ アプリ名「ラリーIQ」、ゲーム、無料。
2. 「アプリの設定」の各申告を `store/listing.md` の回答メモどおりに入力する。
3. 「メインのストアの掲載情報」に `store/listing.md` の文面と `store/` の画像を入れる。
4. **パッケージ名は最初のアップロードで永久に固定**されます。現在は `io.github.seitoshiki.rallyiq`。変える場合は、アップロード前に `capacitor.config.json` の `appId` と `android/app/build.gradle` の `namespace`・`applicationId`、`android/app/src/main/java/` 以下のフォルダと `MainActivity.java` の package を揃えて変更してください。

### C-5. 内部テストで家族に配る

1. テスト → 内部テスト → テスターのメールリスト（家族の Google アカウント）を作る。
2. リリースを作成して AAB をアップロード。Play App Signing は既定のまま有効にする。
3. 表示される「テスト参加用リンク」を家族に送る → 参加 → Play ストアからインストール。

## D. 一般公開する（個人アカウントの場合）

1. テスト → クローズドテスト にトラックを作り、**12人以上**を14日間連続で参加させる（途中で12人を下回ると数え直しになる、という報告が多いので余裕を持って15人程度を推奨）。家族・部活やサークルの仲間に声をかけると集めやすいです。
2. 14日経ったら「製品版へのアクセスを申請」から、テストの内容と得られたフィードバックを回答する。
3. 承認後、製品版トラックにリリースを作成して審査に出す。

## アップデートのしかた

1. `index.html` / `engine.js` を直す。
2. `package.json` の `version` を上げる（例: 1.0.0 → 1.1.0）。
3. push → Web 版は即反映。Android はワークフローの AAB を Play Console の各トラックにアップロード。

## 参考: 要件の出典

- ターゲット API レベル: 2026年8月31日以降の新規アプリと更新は Android 16（API 36）以上が必須。このプロジェクトは `targetSdkVersion = 36`。
- 12人×14日のクローズドテスト要件: Play Console ヘルプ「新しい個人デベロッパー アカウントのアプリのテスト要件」を、申請前に最新版で確認してください。
