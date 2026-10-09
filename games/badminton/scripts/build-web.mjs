#!/usr/bin/env node
// index.html（Artifact 用の本文のみの HTML）から、単体で動く配布用 Web 一式を作る。
//   node scripts/build-web.mjs [--out dist] [--no-fonts]
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

await rm(out, { recursive: true, force: true });
await mkdir(out, { recursive: true });

let src = await readFile(path.join(root, 'index.html'), 'utf8');
const split = src.indexOf('<div class="app">');
if (split < 0) throw new Error('index.html に <div class="app"> が見つからない');
let head = src.slice(0, split).trim();
const body = src.slice(split).trim();

// ---- フォント同梱 ----
const fontFiles = [];
if (withFonts) {
  const m = head.match(/<link rel="stylesheet" href="(https:\/\/fonts\.googleapis\.com\/[^"]+)">/);
  try {
    if (!m) throw new Error('Google Fonts の link がない');
    const cssUrl = m[1].replace(/&amp;/g, '&');
    const ua = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36';
    const res = await fetch(cssUrl, { headers: { 'User-Agent': ua } });
    if (!res.ok) throw new Error('fonts css ' + res.status);
    let css = await res.text();
    const urls = [...new Set(css.match(/https:\/\/fonts\.gstatic\.com\/[^)]+/g) || [])];
    await mkdir(path.join(out, 'fonts'), { recursive: true });
    await Promise.all(urls.map(async (u) => {
      const r = await fetch(u);
      if (!r.ok) throw new Error('font ' + r.status + ' ' + u);
      const buf = Buffer.from(await r.arrayBuffer());
      const name = createHash('sha1').update(u).digest('hex').slice(0, 16) + path.extname(new URL(u).pathname);
      await writeFile(path.join(out, 'fonts', name), buf);
      fontFiles.push('fonts/' + name);
      css = css.split(u).join(name);
    }));
    await writeFile(path.join(out, 'fonts', 'fonts.css'), css);
    head = head
      .replace(m[0], '<link rel="stylesheet" href="fonts/fonts.css">')
      .replace(/<link rel="preconnect"[^>]*>\s*/g, '');
    console.log(`fonts: ${urls.length} files bundled`);
  } catch (e) {
    console.warn('fonts: 同梱をスキップ（CDN 参照のまま）:', e.message);
  }
}

// ---- HTML ----
const html = `<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#0d1719">
<meta name="description" content="${pkg.description}">
<link rel="manifest" href="manifest.webmanifest">
<link rel="icon" type="image/png" sizes="192x192" href="icons/icon-192.png">
<link rel="apple-touch-icon" href="icons/apple-touch-icon.png">
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
    navigator.serviceWorker.register('sw.js').catch(() => {});
  }
</script>
</body>
</html>
`;
await writeFile(path.join(out, 'index.html'), html);

// ---- 静的ファイル ----
await cp(path.join(root, 'engine.js'), path.join(out, 'engine.js'));
await cp(path.join(root, 'privacy.html'), path.join(out, 'privacy.html'));
await cp(path.join(root, 'assets', 'icons'), path.join(out, 'icons'), { recursive: true });

const manifest = {
  name: 'ラリーIQ — バドミントン配球バトル',
  short_name: 'ラリーIQ',
  description: pkg.description,
  lang: 'ja',
  start_url: './',
  scope: './',
  display: 'standalone',
  orientation: 'any',
  background_color: '#0d1719',
  theme_color: '#0d1719',
  categories: ['games', 'sports'],
  icons: [
    { src: 'icons/icon-192.png', sizes: '192x192', type: 'image/png' },
    { src: 'icons/icon-512.png', sizes: '512x512', type: 'image/png' },
    { src: 'icons/icon-maskable-512.png', sizes: '512x512', type: 'image/png', purpose: 'maskable' },
  ],
};
await writeFile(path.join(out, 'manifest.webmanifest'), JSON.stringify(manifest, null, 2));

// フォント（数百の分割ファイル）は初回に全部は取らず、使われた分だけ実行時にキャッシュする
const precache = ['./', 'index.html', 'engine.js', 'privacy.html', 'manifest.webmanifest',
  'icons/icon-192.png', 'icons/icon-512.png', 'icons/apple-touch-icon.png',
  ...(fontFiles.length ? ['fonts/fonts.css'] : [])];
const version = createHash('sha1').update(html + JSON.stringify(precache) + fontFiles.join() + pkg.version).digest('hex').slice(0, 10);
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
