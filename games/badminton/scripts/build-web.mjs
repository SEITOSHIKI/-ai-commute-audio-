#!/usr/bin/env node
// Artifact 用の本文のみの HTML から、単体で動く配布用 Web 一式を作る。
//   node scripts/build-web.mjs [--out dist] [--no-fonts]
//
//   dist/index.html        … 3D 対戦「ラリーIQ ARENA」（arena/）
//   dist/drill/index.html  … 2D 配球ドリル（index.html + engine.js）
//
// - <!doctype>/<head> を補い、セーフエリアと PWA（manifest / service worker）を追加
// - Google Fonts をダウンロードして同梱（アプリ内でオフラインでも同じ書体）。失敗時は CDN 参照のまま
// 依存パッケージなし（Node 18+）。
import { readFile, writeFile, mkdir, cp, rm } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const outArg = args.includes('--out') ? args[args.indexOf('--out') + 1] : 'dist';
const out = path.resolve(process.cwd(), outArg);
const withFonts = !args.includes('--no-fonts');
const pkg = JSON.parse(await readFile(path.join(root, 'package.json'), 'utf8'));
const FONT_LINK = /<link rel="stylesheet" href="(https:\/\/fonts\.googleapis\.com\/[^"]+)">/;

await rm(out, { recursive: true, force: true });
await mkdir(out, { recursive: true });

// ---- フォント同梱（両ページ共通） ----
const fontFiles = [];
let fontsBundled = false;
async function bundleFonts(cssUrl) {
  const ua = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36';
  const res = await fetch(cssUrl, { headers: { 'User-Agent': ua } });
  if (!res.ok) throw new Error('fonts css ' + res.status);
  let css = await res.text();
  const urls = [...new Set(css.match(/https:\/\/fonts\.gstatic\.com\/[^)]+/g) || [])];
  await mkdir(path.join(out, 'fonts'), { recursive: true });
  await Promise.all(urls.map(async (u) => {
    const r = await fetch(u);
    if (!r.ok) throw new Error('font ' + r.status + ' ' + u);
    const name = createHash('sha1').update(u).digest('hex').slice(0, 16) + path.extname(new URL(u).pathname);
    await writeFile(path.join(out, 'fonts', name), Buffer.from(await r.arrayBuffer()));
    fontFiles.push('fonts/' + name);
    css = css.split(u).join(name);
  }));
  await writeFile(path.join(out, 'fonts', 'fonts.css'), css);
  console.log(`fonts: ${urls.length} files bundled`);
}
if (withFonts) {
  const src = await readFile(path.join(root, 'arena', 'index.html'), 'utf8');
  const m = src.match(FONT_LINK);
  try {
    if (!m) throw new Error('Google Fonts の link がない');
    await bundleFonts(m[1].replace(/&amp;/g, '&'));
    fontsBundled = true;
  } catch (e) {
    console.warn('fonts: 同梱をスキップ（CDN 参照のまま）:', e.message);
  }
}

// ---- ページを単体 HTML に包む ----
// up: ページからルートへの相対パス（'' か '../'）
function wrap(src, marker, up, inject = s => s) {
  const split = src.indexOf(marker);
  if (split < 0) throw new Error(`${marker} が見つからない`);
  let head = src.slice(0, split).trim();
  const body = inject(src.slice(split).trim());
  if (fontsBundled) {
    head = head.replace(FONT_LINK, `<link rel="stylesheet" href="${up}fonts/fonts.css">`)
      .replace(/<link rel="preconnect"[^>]*>\s*/g, '');
  }
  return `<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#0b1416">
<meta name="description" content="${pkg.description}">
<link rel="manifest" href="${up}manifest.webmanifest">
<link rel="icon" type="image/png" sizes="192x192" href="${up}icons/icon-192.png">
<link rel="apple-touch-icon" href="${up}icons/apple-touch-icon.png">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<style>
  :root { box-sizing: border-box; padding-top: env(safe-area-inset-top, 0px); padding-bottom: env(safe-area-inset-bottom, 0px); }
  body { margin: 0; font: 14px system-ui, sans-serif; -webkit-tap-highlight-color: transparent; -webkit-user-select: none; user-select: none; }
  img { max-width: 100%; }
  [hidden] { display: none !important; }
</style>
${head}
</head>
<body>
${body}
<script>
  // Web 版のみ: オフライン用 service worker（Android アプリ内では不要）
  if ('serviceWorker' in navigator && location.protocol === 'https:' && !window.Capacitor) {
    navigator.serviceWorker.register('${up}sw.js').catch(() => {});
  }
</script>
</body>
</html>
`;
}

// 3D アリーナ（ルート）。タイトルに 2D ドリルへの入口を足す
const arenaSrc = await readFile(path.join(root, 'arena', 'index.html'), 'utf8');
const arenaHtml = wrap(arenaSrc, '<div id="view">', '', b => b.replace(
  '<div class="rotate">',
  '<a class="ghost" href="drill/" style="text-align:center;text-decoration:none">2D 配球ドリル（ゾーンを選ぶだけの判断練習）</a>\n    <div class="rotate">'));
await writeFile(path.join(out, 'index.html'), arenaHtml);
for (const f of ['game.js', 'physics.js', 'match.js', 'roster.js']) await cp(path.join(root, 'arena', f), path.join(out, f));
await cp(path.join(root, 'arena', 'vendor'), path.join(out, 'vendor'), { recursive: true });

// 2D ドリル
const drillSrc = await readFile(path.join(root, 'index.html'), 'utf8');
const drillHtml = wrap(drillSrc, '<div class="app">', '../', b => b.replace(
  '<button class="icon-btn" id="soundBtn"',
  '<a class="icon-btn" href="../" style="text-decoration:none">3D ARENA</a>\n    <button class="icon-btn" id="soundBtn"'));
await mkdir(path.join(out, 'drill'), { recursive: true });
await writeFile(path.join(out, 'drill', 'index.html'), drillHtml);
await cp(path.join(root, 'engine.js'), path.join(out, 'drill', 'engine.js'));

// ---- 静的ファイル ----
await cp(path.join(root, 'privacy.html'), path.join(out, 'privacy.html'));
await cp(path.join(root, 'assets', 'icons'), path.join(out, 'icons'), { recursive: true });

const manifest = {
  name: 'ラリーIQ — バドミントン配球バトル',
  short_name: 'ラリーIQ',
  description: pkg.description,
  lang: 'ja',
  start_url: './',
  scope: './',
  display: 'fullscreen',
  orientation: 'landscape',
  background_color: '#0b1416',
  theme_color: '#0b1416',
  categories: ['games', 'sports'],
  icons: [
    { src: 'icons/icon-192.png', sizes: '192x192', type: 'image/png' },
    { src: 'icons/icon-512.png', sizes: '512x512', type: 'image/png' },
    { src: 'icons/icon-maskable-512.png', sizes: '512x512', type: 'image/png', purpose: 'maskable' },
  ],
};
await writeFile(path.join(out, 'manifest.webmanifest'), JSON.stringify(manifest, null, 2));

// フォント（数百の分割ファイル）は初回に全部は取らず、使われた分だけ実行時にキャッシュする
const precache = ['./', 'index.html', 'game.js', 'physics.js', 'match.js', 'roster.js', 'vendor/three.module.js', 'vendor/three.core.js',
  'drill/', 'drill/index.html', 'drill/engine.js', 'privacy.html', 'manifest.webmanifest',
  'icons/icon-192.png', 'icons/icon-512.png', 'icons/apple-touch-icon.png',
  ...(fontsBundled ? ['fonts/fonts.css'] : [])];
const hashSrc = arenaHtml + drillHtml + await readFile(path.join(out, 'game.js'), 'utf8') +
  await readFile(path.join(out, 'match.js'), 'utf8') + await readFile(path.join(out, 'physics.js'), 'utf8');
const version = createHash('sha1').update(hashSrc + JSON.stringify(precache) + fontFiles.join() + pkg.version).digest('hex').slice(0, 10);
const sw = `// 自動生成（scripts/build-web.mjs）
const CACHE = 'rallyiq-${version}';
const FILES = ${JSON.stringify(precache)};
self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(FILES)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET') return;
  e.respondWith(caches.match(e.request).then(hit => hit || fetch(e.request).then(res => {
    if (res.ok && new URL(e.request.url).origin === location.origin) {
      const copy = res.clone();
      caches.open(CACHE).then(c => c.put(e.request, copy));
    }
    return res;
  })));
});
`;
await writeFile(path.join(out, 'sw.js'), sw);
console.log(`built → ${path.relative(process.cwd(), out) || '.'} (v${pkg.version}, cache ${version})`);
