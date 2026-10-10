// ラリーIQ アリーナ — 3D 描画・操作・選手選択と育成。試合のルールと物理は match.js / physics.js。
import * as THREE from './vendor/three.module.js';
import { C, LEVELS } from './physics.js';
import { Match } from './match.js';
import {
  CHARACTERS, STAT_KEYS, STAT_JP, STAT_HINT, getCharacter, toAttr, awardMatch, allocate, rename, overall, xpToNext, costFor,
} from './roster.js';

const $ = id => document.getElementById(id);
const store = {
  get(k) { try { return JSON.parse(localStorage.getItem(k)); } catch (e) { return null; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* 保存できなくても遊べる */ } },
};
const COL = { you: 0xf4b63f, cpu: 0xee6c5a, line: 0xf1f4ec };
const GUIDE = { beginner: 1, club: 0.7, expert: 0.4, pro: 0.22 };
const FACE_NAMES = { '-2': 'リバース強', '-1': 'リバース', '0': 'フラット', '1': 'カット', '2': 'カット強' };
const H_NAMES = { down: '沈める', flat: '低く速く', mid: 'ふつう', high: '高く' };
const fine = matchMedia('(pointer: fine)').matches;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const smooth = t => t * t * (3 - 2 * t);

// =====================================================================
// シーン
// =====================================================================
const view = $('view');
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.08;
view.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b1416);
scene.fog = new THREE.Fog(0x0b1416, 26, 60);
const camera = new THREE.PerspectiveCamera(46, 1, 0.1, 200);

scene.add(new THREE.HemisphereLight(0xe8f2ec, 0x1a2a2c, 1.05));
const key = new THREE.DirectionalLight(0xffffff, 1.6); key.position.set(-6, 14, 7); scene.add(key);
const rim = new THREE.DirectionalLight(0xbfd8ff, 0.55); rim.position.set(9, 6, -8); scene.add(rim);

function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.fov = camera.aspect >= 1.3 ? 46 : camera.aspect >= 1 ? 52 : 70;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);

// ---- 床・コート・ネット ----
function courtTexture() {
  const S = 100, W = 14.6, H = 7.3;
  const c = document.createElement('canvas'); c.width = W * S; c.height = H * S;
  const g = c.getContext('2d');
  g.fillStyle = '#1f5e4c'; g.fillRect(0, 0, c.width, c.height);
  g.strokeStyle = 'rgba(0,0,0,0.08)'; g.lineWidth = 2;
  for (let z = -3; z <= 3; z += 1.5) { g.beginPath(); g.moveTo(0, (z + H / 2) * S); g.lineTo(c.width, (z + H / 2) * S); g.stroke(); }
  g.strokeStyle = '#f1f4ec'; g.lineWidth = 4;
  const L = (x0, z0, x1, z1) => { g.beginPath(); g.moveTo((x0 + W / 2) * S, (z0 + H / 2) * S); g.lineTo((x1 + W / 2) * S, (z1 + H / 2) * S); g.stroke(); };
  L(-6.7, -3.05, 6.7, -3.05); L(-6.7, 3.05, 6.7, 3.05); L(-6.7, -3.05, -6.7, 3.05); L(6.7, -3.05, 6.7, 3.05);
  L(-6.7, -C.W, 6.7, -C.W); L(-6.7, C.W, 6.7, C.W);
  for (const s of [-1, 1]) { L(s * C.SHORT, -3.05, s * C.SHORT, 3.05); L(s * 5.94, -3.05, s * 5.94, 3.05); L(s * C.SHORT, 0, s * 6.7, 0); }
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = renderer.capabilities.getMaxAnisotropy();
  return t;
}
const floor = new THREE.Mesh(new THREE.PlaneGeometry(80, 50), new THREE.MeshStandardMaterial({ color: 0x132326, roughness: 0.85 }));
floor.rotation.x = -Math.PI / 2; scene.add(floor);
const court = new THREE.Mesh(new THREE.PlaneGeometry(14.6, 7.3), new THREE.MeshStandardMaterial({ map: courtTexture(), roughness: 0.7 }));
court.rotation.x = -Math.PI / 2; court.position.y = 0.004; scene.add(court);
const serveGlow = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ color: COL.line, transparent: true, opacity: 0.12, depthWrite: false }));
serveGlow.rotation.x = -Math.PI / 2; serveGlow.position.y = 0.008; serveGlow.visible = false; scene.add(serveGlow);

function netTexture() {
  const c = document.createElement('canvas'); c.width = 512; c.height = 64;
  const g = c.getContext('2d'); g.strokeStyle = 'rgba(20,24,24,0.9)'; g.lineWidth = 1.2;
  for (let x = 0; x <= 512; x += 6) { g.beginPath(); g.moveTo(x, 0); g.lineTo(x, 64); g.stroke(); }
  for (let y = 0; y <= 64; y += 6) { g.beginPath(); g.moveTo(0, y); g.lineTo(512, y); g.stroke(); }
  const t = new THREE.CanvasTexture(c); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(4, 1); return t;
}
const net = new THREE.Mesh(new THREE.PlaneGeometry(6.1, 0.76), new THREE.MeshBasicMaterial({ map: netTexture(), transparent: true, side: THREE.DoubleSide, depthWrite: false }));
net.rotation.y = Math.PI / 2; net.position.set(0, C.NET_H - 0.38, 0); scene.add(net);
const tape = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.05, 6.1), new THREE.MeshStandardMaterial({ color: 0xf6f7f2 }));
tape.position.set(0, C.NET_H - 0.025, 0); scene.add(tape);
for (const z of [-3.12, 3.12]) {
  const post = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.05, 1.55, 12), new THREE.MeshStandardMaterial({ color: 0xc9d2cf, metalness: 0.4, roughness: 0.4 }));
  post.position.set(0, 0.775, z); scene.add(post);
}
// 奥の壁と照明
const wall = new THREE.Mesh(new THREE.PlaneGeometry(50, 18), new THREE.MeshBasicMaterial({ color: 0x0f1c1f }));
wall.position.set(15, 7, 0); wall.rotation.y = -Math.PI / 2; scene.add(wall);
for (let i = -4; i <= 4; i++) {
  const strip = new THREE.Mesh(new THREE.PlaneGeometry(4.2, 0.14), new THREE.MeshBasicMaterial({ color: i % 2 === 0 ? 0xf4b63f : 0x3a5a5c }));
  strip.position.set(14.9, 2.6, i * 5); strip.rotation.y = -Math.PI / 2; scene.add(strip);
}
for (let i = 0; i < 3; i++) for (const s of [-1, 1]) {
  const cone = new THREE.Mesh(new THREE.ConeGeometry(2.6, 14, 24, 1, true), new THREE.MeshBasicMaterial({ color: 0xfaf4e0, transparent: true, opacity: 0.03, depthWrite: false, side: THREE.DoubleSide, blending: THREE.AdditiveBlending }));
  cone.position.set(s * (2.5 + i * 4), 7, (i - 1) * 4.5); scene.add(cone);
}
// 相手コートの狙いグリッド（前・ハーフ・奥 × 左・センター・右）
{
  const pts = [];
  for (const x of [2.3, 4.6]) pts.push(new THREE.Vector3(x, 0.012, -C.W), new THREE.Vector3(x, 0.012, C.W));
  for (const z of [-0.86, 0.86]) pts.push(new THREE.Vector3(0.05, 0.012, z), new THREE.Vector3(C.L, 0.012, z));
  const grid = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.14 }));
  scene.add(grid);
}

const blobMat = new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.32, depthWrite: false });
function blob(r) { const m = new THREE.Mesh(new THREE.CircleGeometry(r, 24), blobMat.clone()); m.rotation.x = -Math.PI / 2; m.position.y = 0.01; scene.add(m); return m; }

// =====================================================================
// 人体モデル（関節ごとの階層。左利きは体ごと鏡像にする）
// =====================================================================
const lathe = (profile, mat, seg = 14) => new THREE.Mesh(new THREE.LatheGeometry(profile.map(([r, y]) => new THREE.Vector2(r, y)), seg), mat);
const sphere = (r, mat, sx = 1, sy = 1, sz = 1) => { const m = new THREE.Mesh(new THREE.SphereGeometry(r, 18, 14), mat); m.scale.set(sx, sy, sz); return m; };
const cyl = (r0, r1, h, mat, seg = 12) => new THREE.Mesh(new THREE.CylinderGeometry(r0, r1, h, seg), mat);

function makeHuman(ch, marker) {
  const mat = c => new THREE.MeshStandardMaterial({ color: c, roughness: 0.62 });
  const M = { skin: mat(ch.skin), shirt: mat(ch.shirt), shorts: mat(ch.shorts), hair: mat(ch.hair), white: mat(0xf3f4f0), dark: mat(0x1a1d1f), lip: mat(0x9a5446) };
  M.hair.roughness = 0.85;
  const wf = 0.88 + 0.3 * ch.build;
  const root = new THREE.Group();
  const body = new THREE.Group(); root.add(body);
  body.scale.set(ch.height / 1.75, ch.height / 1.75, (ch.height / 1.75) * (ch.lefty ? -1 : 1));

  const hips = new THREE.Group(); hips.position.y = 0.94; body.add(hips);
  const pelvis = lathe([[0.001, 0.03], [0.128, 0.03], [0.148, -0.05], [0.142, -0.13], [0.095, -0.19], [0.001, -0.2]], M.shorts);
  pelvis.scale.set(0.78, 1, wf); hips.add(pelvis);

  const spine = new THREE.Group(); spine.position.y = 0.03; hips.add(spine);
  const torso = lathe([[0.001, 0], [0.128, 0], [0.13, 0.1], [0.15, 0.24], [0.168, 0.36], [0.16, 0.44], [0.11, 0.5], [0.055, 0.535], [0.001, 0.54]], M.shirt, 20);
  torso.scale.set(0.68, 1, wf); spine.add(torso);
  const chestBand = cyl(0.135, 0.135, 0.03, M.white, 20); chestBand.scale.set(0.69, 1, wf); chestBand.position.y = 0.3; spine.add(chestBand);

  const neckG = new THREE.Group(); neckG.position.y = 0.52; spine.add(neckG);
  const neck = cyl(0.043, 0.05, 0.1, M.skin); neck.position.y = 0.04; neckG.add(neck);
  const head = new THREE.Group(); head.position.y = 0.09; neckG.add(head);
  const skull = sphere(0.1, M.skin, 1.06, 1.18, 0.93); skull.position.y = 0.1; head.add(skull);
  const jaw = sphere(0.074, M.skin, 1.05, 0.8, 0.88); jaw.position.set(0.022, 0.04, 0); head.add(jaw);
  for (const s of [-1, 1]) {
    const ear = sphere(0.022, M.skin, 0.5, 1, 0.4); ear.position.set(-0.005, 0.1, s * 0.092); head.add(ear);
    const eye = sphere(0.0115, M.dark); eye.position.set(0.093, 0.112, s * 0.034); head.add(eye);
    const brow = new THREE.Mesh(new THREE.BoxGeometry(0.01, 0.008, 0.036), M.hair); brow.position.set(0.096, 0.14, s * 0.035); head.add(brow);
  }
  const nose = new THREE.Mesh(new THREE.ConeGeometry(0.014, 0.036, 8), M.skin); nose.rotation.z = -Math.PI / 2; nose.position.set(0.112, 0.092, 0); head.add(nose);
  const mouth = new THREE.Mesh(new THREE.BoxGeometry(0.006, 0.006, 0.032), M.lip); mouth.position.set(0.094, 0.058, 0); head.add(mouth);
  // 髪型
  const cap = (scale, thetaLen) => {
    const m = new THREE.Mesh(new THREE.SphereGeometry(0.106, 20, 12, 0, Math.PI * 2, 0, thetaLen), M.hair);
    m.scale.set(1.08 * scale, 1.15 * scale, 0.98 * scale); m.position.set(-0.006, 0.112, 0); m.rotation.z = 0.32; head.add(m); return m;
  };
  if (ch.hairStyle === 'buzz') cap(1.0, Math.PI * 0.5);
  else {
    cap(1.03, Math.PI * 0.56);
    if (ch.hairStyle === 'spiky') for (let i = 0; i < 7; i++) {
      const sp = new THREE.Mesh(new THREE.ConeGeometry(0.022, 0.07, 6), M.hair);
      const a = (i / 7) * Math.PI * 2; sp.position.set(Math.cos(a) * 0.045 - 0.01, 0.215, Math.sin(a) * 0.045); sp.rotation.set(Math.sin(a) * 0.5, 0, -Math.cos(a) * 0.5); head.add(sp);
    }
    if (ch.hairStyle === 'swept') { const f = sphere(0.06, M.hair, 1.2, 0.5, 1.5); f.position.set(0.07, 0.19, 0.02); f.rotation.z = -0.4; head.add(f); }
    if (ch.hairStyle === 'pony') { const b = sphere(0.04, M.hair); b.position.set(-0.11, 0.13, 0); head.add(b); const t = cyl(0.022, 0.012, 0.13, M.hair); t.position.set(-0.13, 0.06, 0); t.rotation.z = -0.3; head.add(t); }
  }

  // 腕（D: 利き腕 = 右側 +z, F: 反対の腕）
  const arm = (zs) => {
    const sh = new THREE.Group(); sh.position.set(0, 0.45, zs * 0.198 * wf); spine.add(sh);
    sh.add(sphere(0.064, M.shirt));
    sh.add(lathe([[0.001, 0], [0.05, -0.02], [0.054, -0.1], [0.046, -0.22], [0.037, -0.29], [0.001, -0.3]], M.skin));
    sh.add(lathe([[0.066, 0.02], [0.066, -0.11], [0.001, -0.115]], M.shirt));
    const el = new THREE.Group(); el.position.y = -0.29; sh.add(el);
    el.add(sphere(0.036, M.skin));
    el.add(lathe([[0.001, 0], [0.038, -0.02], [0.042, -0.07], [0.031, -0.2], [0.024, -0.26], [0.001, -0.27]], M.skin));
    const wr = new THREE.Group(); wr.position.y = -0.265; el.add(wr);
    const hand = sphere(0.04, M.skin, 0.72, 1.2, 0.55); hand.position.y = -0.045; wr.add(hand);
    const band = cyl(0.03, 0.03, 0.04, M.white); band.position.y = 0.012; wr.add(band);
    return { sh, el, wr };
  };
  const D = arm(1), F = arm(-1);
  // ラケット（利き手に）
  const racket = new THREE.Group(); racket.position.y = -0.06; D.wr.add(racket);
  const grip = cyl(0.015, 0.014, 0.2, mat(ch.shirt)); grip.position.y = -0.06; racket.add(grip);
  const shaft = cyl(0.005, 0.004, 0.27, M.dark, 6); shaft.position.y = -0.29; racket.add(shaft);
  const frame = new THREE.Mesh(new THREE.TorusGeometry(0.11, 0.0065, 6, 32), M.dark);
  frame.scale.set(1, 1.3, 1); frame.position.y = -0.56; frame.rotation.y = Math.PI / 2; racket.add(frame);
  const strings = new THREE.Mesh(new THREE.CircleGeometry(0.106, 24), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.2, side: THREE.DoubleSide }));
  strings.scale.set(1, 1.3, 1); strings.position.y = -0.56; strings.rotation.y = Math.PI / 2; racket.add(strings);

  // 脚
  const leg = (zs) => {
    const hp = new THREE.Group(); hp.position.set(0, -0.08, zs * 0.088 * wf); hips.add(hp);
    hp.add(lathe([[0.001, 0.02], [0.084, -0.03], [0.088, -0.12], [0.07, -0.32], [0.053, -0.43], [0.001, -0.44]], M.skin));
    hp.add(lathe([[0.097, 0.04], [0.097, -0.21], [0.001, -0.215]], M.shorts));
    const kn = new THREE.Group(); kn.position.y = -0.435; hp.add(kn);
    kn.add(sphere(0.05, M.skin));
    kn.add(lathe([[0.001, 0], [0.05, -0.02], [0.058, -0.1], [0.045, -0.25], [0.032, -0.37], [0.001, -0.41]], M.skin));
    const sock = cyl(0.036, 0.034, 0.1, M.white); sock.position.y = -0.36; kn.add(sock);
    const an = new THREE.Group(); an.position.y = -0.42; kn.add(an);
    const shoe = sphere(0.1, M.white, 1.32, 0.42, 0.52); shoe.position.set(0.05, -0.03, 0); an.add(shoe);
    const sole = new THREE.Mesh(new THREE.BoxGeometry(0.26, 0.022, 0.095), mat(ch.shirt)); sole.position.set(0.05, -0.065, 0); an.add(sole);
    return { hp, kn, an };
  };
  const LD = leg(1), LF = leg(-1);

  const ring = new THREE.Mesh(new THREE.RingGeometry(0.42, 0.47, 40), new THREE.MeshBasicMaterial({ color: marker, transparent: true, opacity: 0.7, depthWrite: false }));
  ring.rotation.x = -Math.PI / 2; ring.position.y = 0.012;
  scene.add(root); scene.add(ring);
  const shadow = blob(0.38);
  const J = { hips, spine, neck: neckG, head, shD: D.sh, elD: D.el, wrD: D.wr, shF: F.sh, elF: F.el, wrF: F.wr, hipD: LD.hp, knD: LD.kn, anD: LD.an, hipF: LF.hp, knF: LF.kn, anF: LF.an };
  return { root, body, J, ring, shadow, cur: {}, phase: 0, ch };
}

function disposeHuman(h) {
  if (!h) return;
  scene.remove(h.root); scene.remove(h.ring); scene.remove(h.shadow);
  h.root.traverse(o => { if (o.geometry) o.geometry.dispose(); if (o.material) o.material.dispose(); });
}

// ---- ポーズ（右利きの角度。z: 前後に振る、x: 横に開く、y: ひねり） ----
// spine の z はマイナスで前傾、y はプラスで利き肩が前へ（打ちに行く）
const POSE = {
  ready:     { hipsY: -0.07, spine: [0, 0, -0.18], shD: [-0.35, 0, 1.0], elD: [0, 0, 1.45], wrD: [0, 0, 0.5], shF: [0.35, 0, 0.6], elF: [0, 0, 1.2], hipD: [0, 0, 0.3], knD: [0, 0, -0.55], hipF: [0, 0, 0.3], knF: [0, 0, -0.55] },
  prep_over: { hipsY: -0.1, spine: [0, -0.85, 0.06], shD: [-1.3, 0, 2.3], elD: [0, 0, 2.3], wrD: [0, 0, -0.5], shF: [0.25, 0, 2.6], elF: [0, 0, 0.25], hipD: [0, 0, -0.25], knD: [0, 0, -0.6], hipF: [0, 0, 0.35], knF: [0, 0, -0.35] },
  smash_c:   { hipsY: -0.04, spine: [0, 0.5, -0.38], shD: [-0.3, 0, 3.0], elD: [0, 0, 0.12], wrD: [0, 0, 0.7], shF: [0.45, 0, 0.9], elF: [0, 0, 1.6], hipD: [0, 0, 0.1], knD: [0, 0, -0.3], hipF: [0, 0, -0.1], knF: [0, 0, -0.2] },
  smash_f:   { hipsY: -0.12, spine: [0, 0.75, -0.55], shD: [0.45, 0, 0.9], elD: [0, 0, 0.6], wrD: [0, 0, 0.9], shF: [0.5, 0, 0.4], elF: [0, 0, 1.4], hipD: [0, 0, 0.6], knD: [0, 0, -0.65], hipF: [0, 0, -0.3], knF: [0, 0, -0.4] },
  clear_c:   { hipsY: -0.04, spine: [0, 0.45, -0.1], shD: [-0.4, 0, 3.1], elD: [0, 0, 0.1], wrD: [0, 0, 0.4], shF: [0.45, 0, 1.0], elF: [0, 0, 1.5], hipD: [0, 0, 0.05], knD: [0, 0, -0.3], hipF: [0, 0, -0.05], knF: [0, 0, -0.25] },
  clear_f:   { hipsY: -0.08, spine: [0, 0.55, -0.25], shD: [0.1, 0, 1.8], elD: [0, 0, 0.5], wrD: [0, 0, 0.7], shF: [0.45, 0, 0.6], elF: [0, 0, 1.4], hipD: [0, 0, 0.4], knD: [0, 0, -0.5], hipF: [0, 0, -0.2], knF: [0, 0, -0.35] },
  drop_c:    { hipsY: -0.05, spine: [0, 0.3, -0.2], shD: [-0.45, 0.4, 3.0], elD: [0, 0, 0.2], wrD: [0, 0.8, 0.15], shF: [0.4, 0, 1.0], elF: [0, 0, 1.5], hipD: [0, 0, 0.05], knD: [0, 0, -0.35], hipF: [0, 0, -0.05], knF: [0, 0, -0.3] },
  drop_f:    { hipsY: -0.08, spine: [0, 0.35, -0.3], shD: [-0.1, 0.5, 2.3], elD: [0, 0, 0.55], wrD: [0, 0.95, 0.35], shF: [0.4, 0, 0.7], elF: [0, 0, 1.4], hipD: [0, 0, 0.3], knD: [0, 0, -0.45], hipF: [0, 0, -0.1], knF: [0, 0, -0.35] },
  prep_fh:   { hipsY: -0.12, spine: [0, -0.6, -0.12], shD: [0, -1.35, 1.5], elD: [0, 0, 1.25], wrD: [0, 0, -0.8], shF: [0.5, 0, 1.0], elF: [0, 0, 1.3], hipD: [0, 0, 0.35], knD: [0, 0, -0.7], hipF: [0, 0, 0.2], knF: [0, 0, -0.5] },
  drive_c:   { hipsY: -0.1, spine: [0, 0.2, -0.15], shD: [0, -0.3, 1.5], elD: [0, 0, 0.3], wrD: [0, 0, 0.25], shF: [0.5, 0, 0.8], elF: [0, 0, 1.4], hipD: [0, 0, 0.35], knD: [0, 0, -0.6], hipF: [0, 0, 0.2], knF: [0, 0, -0.5] },
  drive_f:   { hipsY: -0.1, spine: [0, 0.55, -0.15], shD: [0, 0.65, 1.5], elD: [0, 0, 0.9], wrD: [0, 0, 0.6], shF: [0.5, 0, 0.6], elF: [0, 0, 1.4], hipD: [0, 0, 0.35], knD: [0, 0, -0.6], hipF: [0, 0, 0.2], knF: [0, 0, -0.5] },
  prep_bh:   { hipsY: -0.12, spine: [0, 0.75, -0.12], shD: [0, 1.3, 1.5], elD: [0, 0, 2.0], wrD: [0, 0, -0.6], shF: [0.5, 0, 0.6], elF: [0, 0, 1.4], hipD: [0, 0, 0.2], knD: [0, 0, -0.5], hipF: [0, 0, 0.35], knF: [0, 0, -0.7] },
  bh_c:      { hipsY: -0.1, spine: [0, 0.1, -0.15], shD: [0, 0.2, 1.5], elD: [0, 0, 0.4], wrD: [0, 0, 0.3], shF: [0.5, 0, 0.6], elF: [0, 0, 1.4], hipD: [0, 0, 0.2], knD: [0, 0, -0.5], hipF: [0, 0, 0.35], knF: [0, 0, -0.65] },
  bh_f:      { hipsY: -0.1, spine: [0, -0.25, -0.15], shD: [0, -0.55, 1.6], elD: [0, 0, 0.3], wrD: [0, 0, 0.6], shF: [0.5, 0, 0.6], elF: [0, 0, 1.4], hipD: [0, 0, 0.2], knD: [0, 0, -0.5], hipF: [0, 0, 0.35], knF: [0, 0, -0.65] },
  push_p:    { hipsY: -0.1, spine: [0, -0.2, -0.28], shD: [0, -0.4, 2.15], elD: [0, 0, 1.8], wrD: [0, 0, -0.6], shF: [0.45, 0, 1.0], elF: [0, 0, 1.2], hipD: [0, 0, 0.6], knD: [0, 0, -0.75], hipF: [0, 0, -0.15], knF: [0, 0, -0.35] },
  push_c:    { hipsY: -0.1, spine: [0, 0.1, -0.32], shD: [0, -0.2, 2.0], elD: [0, 0, 0.3], wrD: [0, 0, 0.45], shF: [0.45, 0, 0.9], elF: [0, 0, 1.2], hipD: [0, 0, 0.65], knD: [0, 0, -0.75], hipF: [0, 0, -0.2], knF: [0, 0, -0.35] },
  push_f:    { hipsY: -0.1, spine: [0, 0.15, -0.32], shD: [0, 0, 1.75], elD: [0, 0, 0.5], wrD: [0, 0, 0.65], shF: [0.45, 0, 0.9], elF: [0, 0, 1.2], hipD: [0, 0, 0.65], knD: [0, 0, -0.75], hipF: [0, 0, -0.2], knF: [0, 0, -0.35] },
  net_fh_p:  { spine: [0, -0.1, -0.45], shD: [-0.2, -0.25, 1.35], elD: [0, 0, 0.25], wrD: [0, 0, -0.35], shF: [0.5, 0, -0.6], elF: [0, 0, 0.4] },
  net_fh_c:  { spine: [0, 0, -0.48], shD: [-0.2, -0.2, 1.45], elD: [0, 0, 0.15], wrD: [0, 0, 0.1], shF: [0.5, 0, -0.6], elF: [0, 0, 0.4] },
  net_fh_f:  { spine: [0, 0, -0.45], shD: [-0.2, -0.15, 1.55], elD: [0, 0, 0.2], wrD: [0, 0, 0.3], shF: [0.5, 0, -0.6], elF: [0, 0, 0.4] },
  net_bh_p:  { spine: [0, 0.5, -0.45], shD: [0, 0.7, 1.35], elD: [0, 0, 0.5], wrD: [0, 0, -0.3], shF: [0.5, 0, -0.6], elF: [0, 0, 0.4] },
  net_bh_c:  { spine: [0, 0.35, -0.48], shD: [0, 0.55, 1.45], elD: [0, 0, 0.2], wrD: [0, 0, 0.1], shF: [0.5, 0, -0.6], elF: [0, 0, 0.4] },
  net_bh_f:  { spine: [0, 0.3, -0.45], shD: [0, 0.45, 1.55], elD: [0, 0, 0.2], wrD: [0, 0, 0.3], shF: [0.5, 0, -0.6], elF: [0, 0, 0.4] },
  lift_fh_p: { spine: [0, -0.35, -0.4], shD: [-0.25, -0.3, -0.5], elD: [0, 0, 0.4], wrD: [0, 0, -0.9], shF: [0.5, 0, -0.5], elF: [0, 0, 0.4] },
  lift_fh_c: { spine: [0, 0, -0.42], shD: [-0.25, -0.2, 1.0], elD: [0, 0, 0.2], wrD: [0, 0, 0.35], shF: [0.5, 0, -0.5], elF: [0, 0, 0.4] },
  lift_fh_f: { spine: [0, 0.3, -0.3], shD: [-0.25, 0, 2.4], elD: [0, 0, 0.6], wrD: [0, 0, 0.9], shF: [0.5, 0, -0.3], elF: [0, 0, 0.4] },
  lift_bh_p: { spine: [0, 0.6, -0.4], shD: [0, 0.8, -0.2], elD: [0, 0, 0.9], wrD: [0, 0, -0.8], shF: [0.5, 0, -0.5], elF: [0, 0, 0.4] },
  lift_bh_c: { spine: [0, 0.35, -0.42], shD: [0, 0.65, 0.9], elD: [0, 0, 0.3], wrD: [0, 0, 0.3], shF: [0.5, 0, -0.5], elF: [0, 0, 0.4] },
  lift_bh_f: { spine: [0, 0.2, -0.3], shD: [0, 0.45, 1.9], elD: [0, 0, 0.4], wrD: [0, 0, 0.8], shF: [0.5, 0, -0.3], elF: [0, 0, 0.4] },
  ss_p:      { hipsY: -0.06, spine: [0, 0.35, -0.15], shD: [0, 0.95, 1.3], elD: [0, 0, 2.0], wrD: [0, 0, -0.35], shF: [0.2, 0, 1.25], elF: [0, 0, 0.35], hipD: [0, 0, 0.4], knD: [0, 0, -0.45], hipF: [0, 0, -0.1], knF: [0, 0, -0.25] },
  ss_c:      { hipsY: -0.06, spine: [0, 0.3, -0.15], shD: [0, 0.65, 1.3], elD: [0, 0, 1.7], wrD: [0, 0, 0.1], shF: [0.3, 0, 0.9], elF: [0, 0, 0.6], hipD: [0, 0, 0.4], knD: [0, 0, -0.45], hipF: [0, 0, -0.1], knF: [0, 0, -0.25] },
  ss_f:      { hipsY: -0.06, spine: [0, 0.25, -0.15], shD: [0, 0.45, 1.4], elD: [0, 0, 1.55], wrD: [0, 0, 0.2], shF: [0.35, 0, 0.7], elF: [0, 0, 1.0], hipD: [0, 0, 0.4], knD: [0, 0, -0.45], hipF: [0, 0, -0.1], knF: [0, 0, -0.25] },
  sl_p:      { hipsY: -0.05, spine: [0, -0.45, -0.12], shD: [-0.3, 0, -0.9], elD: [0, 0, 0.4], wrD: [0, 0, -1.0], shF: [0.3, 0, 1.3], elF: [0, 0, 0.4], hipD: [0, 0, -0.2], knD: [0, 0, -0.3], hipF: [0, 0, 0.35], knF: [0, 0, -0.3] },
  sl_c:      { hipsY: -0.05, spine: [0, 0.1, -0.15], shD: [-0.3, 0, 0.8], elD: [0, 0, 0.3], wrD: [0, 0, 0.4], shF: [0.3, 0, 0.8], elF: [0, 0, 0.7], hipD: [0, 0, 0.1], knD: [0, 0, -0.3], hipF: [0, 0, 0.1], knF: [0, 0, -0.3] },
  sl_f:      { hipsY: -0.05, spine: [0, 0.45, -0.1], shD: [-0.3, 0.5, 2.6], elD: [0, 0, 0.6], wrD: [0, 0, 1.0], shF: [0.3, 0, 0.5], elF: [0, 0, 1.0], hipD: [0, 0, 0.35], knD: [0, 0, -0.3], hipF: [0, 0, -0.1], knF: [0, 0, -0.3] },
};
// ランジ（利き足を大きく前へ）とジャンプの脚
const LUNGE = { hipsY: -0.36, hipD: [0, 0, 1.25], knD: [0, 0, -1.45], anD: [0, 0, 0.3], hipF: [0, 0, -0.55], knF: [0, 0, -0.25], anF: [0, 0, -0.4] };
const JUMP_LEGS = { hipD: [0, 0, 0.5], knD: [0, 0, -1.3], hipF: [0, 0, -0.55], knF: [0, 0, -0.9] };
const SWINGS = {
  smash: ['prep_over', 'smash_c', 'smash_f'], jumpsmash: ['prep_over', 'smash_c', 'smash_f'],
  clear: ['prep_over', 'clear_c', 'clear_f'], drop: ['prep_over', 'drop_c', 'drop_f'],
  drive_fh: ['prep_fh', 'drive_c', 'drive_f'], drive_bh: ['prep_bh', 'bh_c', 'bh_f'],
  push_fh: ['push_p', 'push_c', 'push_f'], push_bh: ['prep_bh', 'bh_c', 'bh_f'],
  net_fh: ['net_fh_p', 'net_fh_c', 'net_fh_f'], net_bh: ['net_bh_p', 'net_bh_c', 'net_bh_f'],
  lift_fh: ['lift_fh_p', 'lift_fh_c', 'lift_fh_f'], lift_bh: ['lift_bh_p', 'lift_bh_c', 'lift_bh_f'],
  serve_short: ['ss_p', 'ss_c', 'ss_f'], serve_long: ['sl_p', 'sl_c', 'sl_f'],
};
const JOINTS = ['spine', 'neck', 'shD', 'elD', 'wrD', 'shF', 'elF', 'wrF', 'hipD', 'knD', 'anD', 'hipF', 'knF', 'anF'];
const swingKey = (type, hand) => (SWINGS[type] ? type : `${type}_${hand}`);

function poseOf(name) { return POSE[name] || POSE.ready; }
function mixPose(a, b, t) {
  const out = { hipsY: (a.hipsY ?? -0.07) * (1 - t) + (b.hipsY ?? -0.07) * t };
  for (const k of JOINTS) {
    const x = a[k] || [0, 0, 0], y = b[k] || [0, 0, 0];
    out[k] = [x[0] + (y[0] - x[0]) * t, x[1] + (y[1] - x[1]) * t, x[2] + (y[2] - x[2]) * t];
  }
  return out;
}
const withLegs = (p, legs, w = 1) => mixPose(p, { ...p, ...legs }, w);

// 打点の予測からどの構えに入るか
function predictPrep(m, id) {
  const p = m.pl[id];
  if (m.phase === 'serve' && m.server === id) return 'serve_short';
  const it = m.ai[id] && m.ai[id].intercept;
  if (!it || m.lastHitter === id || m.phase !== 'rally') return null;
  const facing = -p.side, racketSide = (p.attr.lefty ? -1 : 1) * facing;
  const hand = (it.z - p.z) * racketSide > -0.12 ? 'fh' : 'bh';
  if (it.y >= 2.0) return 'smash';
  if (it.y >= 1.1) return Math.abs(it.x) < 2.4 && it.y > 1.55 ? `push_${hand}` : `drive_${hand}`;
  return Math.abs(it.x) < 2.6 ? `net_${hand}` : `lift_${hand}`;
}

function animateHuman(h, id, m, dt) {
  const p = m.pl[id];
  h.root.position.set(p.x, p.y, p.z);
  h.ring.position.set(p.x, 0.012, p.z);
  h.shadow.position.set(p.x, 0.01, p.z); h.shadow.scale.setScalar(1 - Math.min(0.4, p.y));
  // 向き: 相手コートを向き、横の球には体を開く
  let yaw = p.side < 0 ? 0 : Math.PI;
  const sh = m.shuttle && m.phase === 'rally' ? m.shuttle.p : null;
  if (sh && m.lastHitter !== id) yaw += clamp(Math.atan2(-(sh.z - p.z), Math.abs(sh.x - p.x) + 1.5) * (p.side < 0 ? 1 : -1), -0.6, 0.6) * 0.5;
  h.root.rotation.y += (yaw - h.root.rotation.y) * Math.min(1, dt * 8);

  // 走り
  const sp = Math.hypot(p.vx, p.vz);
  h.phase += dt * (5 + sp * 3.4);
  const k = Math.min(1, sp / 2.8);
  const run = {
    hipD: [0, 0, 0.3 + Math.sin(h.phase) * 0.75 * k], knD: [0, 0, -0.55 - Math.max(0, -Math.cos(h.phase)) * 0.9 * k],
    hipF: [0, 0, 0.3 - Math.sin(h.phase) * 0.75 * k], knF: [0, 0, -0.55 - Math.max(0, Math.cos(h.phase)) * 0.9 * k],
  };

  let target;
  const since = m.time - p.swingT;
  const sw = p.swing;
  if (sw && since >= 0 && since < 0.6) {
    const [, c, f] = SWINGS[swingKey(sw.type, sw.hand)] || SWINGS.drive_fh;
    if (since < 0.05) target = poseOf(c);
    else if (since < 0.3) target = mixPose(poseOf(c), poseOf(f), smooth((since - 0.05) / 0.25));
    else target = mixPose(poseOf(f), POSE.ready, smooth((since - 0.3) / 0.3));
    if (sw.lunge || sw.type === 'net' || sw.type === 'lift') target = withLegs(target, LUNGE, since < 0.4 ? 1 : 1 - (since - 0.4) / 0.2);
    h.snap = since < 0.05;
  } else {
    h.snap = false;
    const prep = predictPrep(m, id);
    const committed = id === 'P' ? !!m.commit : true;
    if (prep && (committed || m.phase === 'serve')) {
      const [pp, c] = SWINGS[prep];
      target = poseOf(pp);
      // 打点の直前に振り出す
      const it = m.ai[id].intercept;
      if (it && m.launchT != null) {
        const tti = m.launchT + it.t - m.time;
        if (tti < 0.12) target = mixPose(target, poseOf(c), clamp(1 - tti / 0.12, 0, 1) * 0.7);
        if (/^(net|lift)/.test(prep) && tti < 0.45) target = withLegs(target, LUNGE, clamp(1 - (tti - 0.1) / 0.35, 0, 1));
      }
      if (!/^(net|lift)/.test(prep)) target = withLegs(target, run, k);
    } else if (prep) {
      target = withLegs(mixPose(POSE.ready, poseOf(SWINGS[prep][0]), 0.35), run, k);
    } else {
      target = withLegs(POSE.ready, run, k);
    }
  }
  if (p.y > 0.05) target = withLegs(target, JUMP_LEGS);
  // 関節へ（素振りの瞬間は追従を速く）
  const rate = h.snap ? 1 : Math.min(1, dt * 16);
  h.cur = h.cur.spine ? mixPose(h.cur, target, rate) : target;
  const J = h.J;
  J.hips.position.y = 0.94 + h.cur.hipsY;
  for (const k2 of JOINTS) { const v = h.cur[k2]; if (v && J[k2]) J[k2].rotation.set(v[0], v[1], v[2]); }
  // 頭はシャトルを見る
  if (sh) {
    const dx = (sh.x - p.x) * -p.side, dy = sh.y - 1.6 * h.ch.height / 1.75;
    J.neck.rotation.z = clamp(Math.atan2(dy, Math.abs(dx) + 0.5), -0.5, 0.8) * 0.6 + 0.15;
  }
}

// =====================================================================
// シャトル・ガイド・照準・エフェクト
// =====================================================================
const shuttle = new THREE.Group();
{
  const cork = new THREE.Mesh(new THREE.SphereGeometry(0.034, 14, 10, 0, Math.PI * 2, 0, Math.PI / 2), new THREE.MeshStandardMaterial({ color: 0xf4b63f, roughness: 0.5 }));
  const skirt = new THREE.Mesh(new THREE.ConeGeometry(0.05, 0.09, 14, 1, true), new THREE.MeshStandardMaterial({ color: 0xffffff, side: THREE.DoubleSide, transparent: true, opacity: 0.95 }));
  skirt.position.y = -0.045; skirt.rotation.x = Math.PI;
  shuttle.add(cork, skirt); shuttle.scale.setScalar(2.2); scene.add(shuttle);
}
const shuttleShadow = blob(0.1);
const dropLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]), new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.18 }));
scene.add(dropLine);
const TRAIL_N = 26;
const trailGeo = new THREE.BufferGeometry(); trailGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(TRAIL_N * 3), 3));
scene.add(new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.5 })));
let trailPts = [];

const GUIDE_N = 500;
const guideGeo = new THREE.BufferGeometry(); guideGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(GUIDE_N * 3), 3));
const guideMat = new THREE.LineDashedMaterial({ color: COL.you, dashSize: 0.18, gapSize: 0.12, transparent: true, opacity: 0.9 });
const guide = new THREE.Line(guideGeo, guideMat); guide.visible = false; scene.add(guide);
const reticle = new THREE.Group();
{
  const ringM = new THREE.MeshBasicMaterial({ color: COL.you, transparent: true, opacity: 0.95, depthWrite: false });
  const r1 = new THREE.Mesh(new THREE.RingGeometry(0.2, 0.26, 32), ringM); r1.rotation.x = -Math.PI / 2;
  const cross = new THREE.Mesh(new THREE.RingGeometry(0.02, 0.05, 16), ringM); cross.rotation.x = -Math.PI / 2;
  const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, 0.6, 6), ringM); pole.position.y = 0.3;
  reticle.add(r1, cross, pole); reticle.position.y = 0.014; reticle.visible = false; scene.add(reticle);
  reticle.userData.mat = ringM;
}

const sparks = [];
const sparkGeo = new THREE.SphereGeometry(0.03, 6, 4);
for (let i = 0; i < 40; i++) { const m = new THREE.Mesh(sparkGeo, new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true })); m.visible = false; scene.add(m); sparks.push({ m, v: new THREE.Vector3(), life: 0 }); }
const shock = new THREE.Mesh(new THREE.RingGeometry(0.2, 0.28, 40), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false }));
scene.add(shock);
let shockT = 9;
function burst(p, color, n, speed) {
  let k = 0;
  for (const s of sparks) {
    if (s.life > 0) continue;
    s.m.visible = true; s.m.position.set(p.x, p.y, p.z); s.m.material.color.setHex(color);
    s.v.set(Math.random() - 0.5, Math.random() * 0.8, Math.random() - 0.5).normalize().multiplyScalar(speed * (0.5 + Math.random()));
    s.life = 0.35 + Math.random() * 0.25;
    if (++k >= n) break;
  }
}

// =====================================================================
// 音
// =====================================================================
let audio = null, soundOn = store.get('arena.sound') !== false;
function initAudio() { if (!audio && soundOn) { try { audio = new (window.AudioContext || window.webkitAudioContext)(); } catch (e) { audio = null; } } }
function noiseHit(power) {
  if (!audio) return;
  const t = audio.currentTime, len = 0.05 + power * 0.08;
  const buf = audio.createBuffer(1, Math.floor(audio.sampleRate * len), audio.sampleRate);
  const d = buf.getChannelData(0);
  for (let i = 0; i < d.length; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / d.length, 3);
  const src = audio.createBufferSource(); src.buffer = buf;
  const f = audio.createBiquadFilter(); f.type = 'bandpass'; f.frequency.value = 1600 + power * 2400; f.Q.value = 1.2;
  const g = audio.createGain(); g.gain.value = 0.2 + power * 0.6;
  src.connect(f).connect(g).connect(audio.destination); src.start(t);
  if (power > 0.8) tone(70, 0.18, 0.25, 'sine');
}
function tone(freq, dur, vol, type = 'triangle') {
  if (!audio) return;
  const t = audio.currentTime;
  const o = audio.createOscillator(); o.type = type; o.frequency.value = freq;
  const g = audio.createGain(); g.gain.setValueAtTime(vol, t); g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  o.connect(g).connect(audio.destination); o.start(t); o.stop(t + dur);
}

// =====================================================================
// 状態
// =====================================================================
const S = {
  screen: 'title', match: null, levelKey: 'club', autoMove: true, target: 11,
  selP: 'ren', selO: null, chP: null, chO: null,
  face: 0, hclass: 'high', zone: 0, zoneT: 0, slow: 1, slowT: 0, freeze: 0, shake: 0,
  aim: null, aimKey: '', pv: null, touchAim: null, stick: null, keys: new Set(),
  oppHitReal: null, decisions: [], nextAt: null, resultAt: null, intro: 0,
};
const humans = { P: null, O: null };
function buildHumans(chP, chO) {
  disposeHuman(humans.P); disposeHuman(humans.O);
  humans.P = makeHuman(chP, COL.you);
  humans.O = makeHuman(chO, COL.cpu);
}

// タイトル背景のデモ（選んだ選手同士）
let demo = null;
function newDemo() {
  const a = getCharacter(S.selP, store), b = getCharacter(S.selO || pickOpponent(), store);
  demo = new Match({ human: false, level: 'expert', levelP: 'expert', attr: { P: toAttr(a), O: toAttr(b) } });
  buildHumans(a, b);
}
function demoTick(dt) {
  if (!demo) return;
  demo.step(dt * 0.55); demo.events.length = 0;
  if (demo.phase === 'dead' && demo.phaseT > 0.6) demo.nextRally();
  if (demo.phase === 'over') { const keep = demo.attr; demo = new Match({ human: false, level: 'expert', levelP: 'expert', attr: keep }); }
}
const cur = () => (S.screen === 'title' ? demo : S.match);

// =====================================================================
// 入力（照準）
// =====================================================================
const raycaster = new THREE.Raycaster();
const ground = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
const ndc = new THREE.Vector2(), hitPt = new THREE.Vector3();
function courtPoint(clientX, clientY) {
  ndc.set((clientX / innerWidth) * 2 - 1, -(clientY / innerHeight) * 2 + 1);
  raycaster.setFromCamera(ndc, camera);
  if (!raycaster.ray.intersectPlane(ground, hitPt)) return null;
  if (hitPt.x < -0.2 || hitPt.x > C.L + 1.2 || Math.abs(hitPt.z) > C.W + 1.2) return null;
  return { x: clamp(hitPt.x, 0.25, C.L + 0.4), z: clamp(hitPt.z, -C.W - 0.4, C.W + 0.4) };
}
function zoneName(t) {
  const lr = t.z < -0.86 ? '左' : t.z > 0.86 ? '右' : 'センター';
  const fb = t.x < 2.3 ? '前' : t.x < 4.6 ? 'ハーフ' : '奥';
  const corner = lr !== 'センター' && fb !== 'ハーフ';
  const out = t.x > C.L || Math.abs(t.z) > C.W;
  return (out ? 'アウト・' : '') + (lr === 'センター' ? `センター${fb}` : `${lr}${fb}`) + (corner && !out ? 'コーナー' : '');
}
const intentNow = () => S.aim && { target: S.aim, hclass: S.hclass, slice: S.face * 0.5 };

const canvas = renderer.domElement;
canvas.addEventListener('pointermove', e => {
  if (S.screen !== 'play') return;
  if (S.stick && e.pointerId === S.stick.id) return updateStick(e);
  if (e.pointerType === 'mouse' || (S.touchAim && e.pointerId === S.touchAim)) S.aim = courtPoint(e.clientX, e.clientY);
});
canvas.addEventListener('pointerdown', e => {
  if (S.screen !== 'play') return;
  initAudio();
  if (e.pointerType !== 'mouse') {
    const st = $('stick');
    if (!st.hidden) {
      const r = st.getBoundingClientRect();
      if (e.clientX < r.right + 30 && e.clientY > r.top - 30) { S.stick = { id: e.pointerId, cx: r.left + r.width / 2, cy: r.top + r.height / 2, vx: 0, vz: 0 }; updateStick(e); canvas.setPointerCapture(e.pointerId); return; }
    }
    S.touchAim = e.pointerId; S.aim = courtPoint(e.clientX, e.clientY);
    canvas.setPointerCapture(e.pointerId);
  }
});
canvas.addEventListener('pointerup', e => {
  if (S.stick && e.pointerId === S.stick.id) { S.stick = null; $('knob').style.transform = ''; return; }
  if (S.screen !== 'play') return;
  if (e.pointerType === 'mouse' && e.button !== 0) return;
  if (e.pointerType !== 'mouse') { if (e.pointerId !== S.touchAim) return; S.aim = courtPoint(e.clientX, e.clientY); S.touchAim = null; }
  commit();
});
canvas.addEventListener('pointercancel', () => { S.touchAim = null; S.stick = null; });
canvas.addEventListener('wheel', e => { if (S.screen === 'play') { e.preventDefault(); stepHeight(e.deltaY > 0 ? -1 : 1); } }, { passive: false });
canvas.addEventListener('contextmenu', e => e.preventDefault());

function updateStick(e) {
  const st = S.stick, R = 50;
  let dx = e.clientX - st.cx, dy = e.clientY - st.cy;
  const l = Math.hypot(dx, dy); if (l > R) { dx *= R / l; dy *= R / l; }
  st.vx = dx / R; st.vz = dy / R;
  $('knob').style.transform = `translate(${dx}px, ${dy}px)`;
}

function commit() {
  const m = S.match; if (!m) return;
  if (!S.aim) { flashHint('相手コートを狙う'); return; }
  if (!m.canCommit()) { flashHint(m.phase === 'serve' ? '相手のサーブを待つ' : '相手が打ってから決める'); return; }
  m.commitShot(intentNow());
  if (S.oppHitReal != null) S.decisions.push((performance.now() - S.oppHitReal) / 1000);
  S.oppHitReal = null;
  tone(1250, 0.05, 0.05);
}

const HORDER = ['down', 'flat', 'mid', 'high'];
function setHeight(h) { S.hclass = h; for (const b of $('heights').children) b.classList.toggle('on', b.dataset.h === h); }
function stepHeight(d) { setHeight(HORDER[clamp(HORDER.indexOf(S.hclass) + d, 0, 3)]); }
for (const b of $('heights').children) b.addEventListener('pointerdown', e => { e.preventDefault(); e.stopPropagation(); setHeight(b.dataset.h); });

window.addEventListener('keydown', e => {
  const k = e.key.toLowerCase();
  if (e.target && e.target.tagName === 'INPUT') return;
  if (S.screen !== 'play') { if (k === 'enter' && S.screen === 'title') startMatch(); return; }
  if ([' ', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright'].includes(k)) e.preventDefault();
  S.keys.add(k);
  if (k >= '1' && k <= '4') setHeight(HORDER[+k - 1]);
  if (k === 'q') setFace(S.face - 1);
  if (k === 'e') setFace(S.face + 1);
  if (k === ' ') S.match && S.match.jump('P');
  if (k === 'shift') useZone();
  if (k === 'escape') toTitle();
});
window.addEventListener('keyup', e => S.keys.delete(e.key.toLowerCase()));
window.addEventListener('blur', () => S.keys.clear());
function moveVector() {
  let x = 0, z = 0;
  if (S.keys.has('w') || S.keys.has('arrowup')) x += 1;
  if (S.keys.has('s') || S.keys.has('arrowdown')) x -= 1;
  if (S.keys.has('a') || S.keys.has('arrowleft')) z -= 1;
  if (S.keys.has('d') || S.keys.has('arrowright')) z += 1;
  if (S.stick) { x += -S.stick.vz; z += S.stick.vx; }
  return { x, z };
}
function setFace(f) {
  S.face = clamp(f, -2, 2);
  $('faceName').textContent = FACE_NAMES[S.face];
  [...$('notches').children].forEach((n, i) => n.classList.toggle('on', i === S.face + 2));
}
function useZone() { if (S.zone < 1 || S.zoneT > 0) return; S.zone = 0; S.zoneT = 3; tone(520, 0.4, 0.08, 'sine'); updateZone(); }
function updateZone() {
  $('zoneBar').firstElementChild.style.width = Math.round(S.zone * 100) + '%';
  $('zoneBar').classList.toggle('full', S.zone >= 1);
  $('bZone').classList.toggle('ready', S.zone >= 1);
}
const press = (id, fn) => $(id).addEventListener('pointerdown', e => { e.preventDefault(); e.stopPropagation(); initAudio(); fn(); });
press('bFaceL', () => setFace(S.face - 1));
press('bFaceR', () => setFace(S.face + 1));
press('bZone', useZone);
press('bJump', () => S.match && S.match.jump('P'));

// =====================================================================
// 試合
// =====================================================================
function pickOpponent() {
  const pool = CHARACTERS.filter(c => c.id !== S.selP && !c.rookie);
  return pool[Math.floor(Math.random() * pool.length)].id;
}
function startMatch() {
  S.levelKey = document.querySelector('input[name="lv"]:checked').value;
  S.autoMove = document.querySelector('input[name="mv"]:checked').value === 'auto';
  S.target = +document.querySelector('input[name="len"]:checked').value;
  store.set('arena.settings', { lv: S.levelKey, mv: S.autoMove ? 'auto' : 'manual', len: S.target, p: S.selP, o: S.selO });
  S.chP = getCharacter(S.selP, store);
  S.chO = getCharacter(S.selO || pickOpponent(), store);
  S.match = new Match({ level: S.levelKey, autoMove: S.autoMove, target: S.target, attr: { P: toAttr(S.chP), O: toAttr(S.chO) } });
  buildHumans(S.chP, S.chO);
  $('nameP').textContent = S.chP.name; $('nameO').textContent = S.chO.name;
  S.decisions = []; S.zone = 0; S.zoneT = 0; setFace(0); setHeight('high'); updateZone();
  S.screen = 'play'; S.nextAt = null; S.resultAt = null; S.oppHitReal = null; S.aim = null; trailPts = [];
  $('titleScreen').hidden = true; $('resultScreen').hidden = true;
  $('touch').hidden = fine; $('stick').hidden = S.autoMove; document.body.classList.toggle('touchui', !fine);
  $('shotcard').hidden = true;
  initAudio();
  updateScore();
  showBanner('READY', `${S.chP.name} vs ${S.chO.name}`, COL.you, 1.0);
  S.intro = 1.4;
  setTimeout(() => { if (S.screen === 'play') showBanner('GO!', '', COL.you, 0.6); }, 1000);
}
function toTitle() {
  S.screen = 'title'; S.match = null; S.aim = null;
  $('resultScreen').hidden = true; $('titleScreen').hidden = false; $('touch').hidden = true;
  hideBanner(); renderSelect(); newDemo();
}

let bannerTimer = null;
function showBanner(text, sub, color, dur) {
  const b = $('banner'); b.hidden = false; b.innerHTML = '';
  b.style.color = '#' + color.toString(16).padStart(6, '0');
  b.append(document.createTextNode(text));
  if (sub) { const s = document.createElement('small'); s.textContent = sub; b.append(s); }
  b.style.animation = 'none'; void b.offsetWidth; b.style.animation = '';
  clearTimeout(bannerTimer); bannerTimer = setTimeout(hideBanner, dur * 1000);
}
function hideBanner() { $('banner').hidden = true; }
let hintFlash = 0;
function flashHint(t) { $('hint').textContent = t; hintFlash = 1.2; }

function updateScore() {
  const m = S.match;
  $('ptsP').textContent = m.score.P; $('ptsO').textContent = m.score.O;
  $('srvP').classList.toggle('on', m.server === 'P'); $('srvO').classList.toggle('on', m.server === 'O');
  $('meta').textContent = `${LEVELS[S.levelKey].jp} · ${S.autoMove ? 'AUTO' : 'MANUAL'} · RALLY ${m.rallyHits}`;
}

function rate(h) {
  if (h.end === 'net') return { cls: 'r-err', text: 'ネットにかかる' };
  if (!h.landIn && !h.serve) return { cls: 'r-err', text: 'アウトの弾道' };
  if (h.serve) return { cls: 'r-ok', text: h.oppHigh > 0.25 ? '高く上げたサーブ。相手は打ち込める' : '低く沈めたサーブ' };
  const m = h.oppMargin, hi = h.oppHigh;
  if (m < 0) return { cls: 'r-ace', text: '相手は届かない！' };
  if (hi >= 0.25) return { cls: 'r-soft', text: `甘い：高い打点で打ち返される（余裕 ${hi.toFixed(2)}秒）` };
  if (m < 0.2) return { cls: 'r-hard', text: `厳しい：相手の余裕 ${m.toFixed(2)}秒` };
  return { cls: 'r-ok', text: hi >= 0 ? `普通：相手の余裕 ${m.toFixed(2)}秒` : '低く沈めた：相手は上げるしかない' };
}

function handleEvents() {
  const m = S.match;
  for (const ev of m.events.splice(0)) {
    if (ev.type === 'hit') {
      const p = m.shuttle.p, power = Math.min(1, ev.kmh / 300);
      noiseHit(power);
      burst(p, ev.by === 'P' ? COL.you : COL.cpu, 6 + Math.round(power * 14), 1.5 + power * 4);
      if (ev.kmh > 230) { S.shake = 0.1 + power * 0.15; S.freeze = 0.05; S.slow = 0.45; S.slowT = 0.25; shock.position.set(p.x, p.y, p.z); shock.lookAt(camera.position); shockT = 0; }
      if (ev.by === 'P') {
        const r = rate(ev);
        $('shotcard').hidden = false;
        $('scName').textContent = ev.name;
        const sl = ev.slice ? ` · ${FACE_NAMES[Math.round(ev.slice / 0.5)]}` : '';
        $('scNums').textContent = `${Math.round(ev.kmh)}km/h · ${ev.elev >= 0 ? '↗' : '↘'}${Math.abs(ev.elev).toFixed(0)}°${sl}`;
        let extra = ev.late ? ' · 判断遅れ' : ev.commitLead > 0.6 ? ' · 早く決めすぎて読まれた' : '';
        if (ev.downgraded) extra += ` · 打点から「${H_NAMES[ev.downgraded]}」は無理`;
        $('scEval').className = 'eval ' + r.cls; $('scEval').textContent = r.text + extra;
        S.zone = Math.min(1, S.zone + (r.cls === 'r-ace' ? 0.34 : r.cls === 'r-hard' ? 0.2 : r.cls === 'r-ok' ? 0.08 : 0));
        updateZone();
        S.oppHitReal = null;
      } else S.oppHitReal = performance.now();
      trailPts = [];
    } else if (ev.type === 'point') {
      updateScore();
      const youWon = ev.winner === 'P', byYou = ev.by === 'P';
      const who = byYou ? S.chP.name : S.chO.name;
      let head, sub;
      if (ev.reason === 'in') { head = youWon ? 'POINT!' : 'LOST'; sub = `${who}の${ev.shot || 'ショット'}${ev.kmh > 200 ? ` ${Math.round(ev.kmh)}km/h` : ''}が決まった`; }
      else if (ev.reason === 'out') { head = 'OUT'; sub = `${who}の${ev.shot || 'ショット'}がアウト`; }
      else { head = 'NET'; sub = `${who}の${ev.shot || 'ショット'}がネット`; }
      showBanner(head, sub, youWon ? COL.you : COL.cpu, 1.5);
      tone(youWon ? 880 : 300, 0.3, 0.1);
      S.slow = 0.35; S.slowT = 0.7; S.nextAt = performance.now() + 1700;
    } else if (ev.type === 'game') {
      S.resultAt = performance.now() + 1900; S.nextAt = null;
      setTimeout(() => showBanner('GAME!', `${ev.score.P} – ${ev.score.O}`, ev.winner === 'P' ? COL.you : COL.cpu, 1.8), 700);
    } else if (ev.type === 'serve-ready') updateScore();
  }
}

// =====================================================================
// ループ
// =====================================================================
let last = performance.now();
function frame(now) {
  const dt = Math.min(0.05, (now - last) / 1000); last = now;
  if (S.screen === 'play' && S.match) stepGame(now, dt);
  else if (S.screen === 'title') demoTick(dt);
  if (S.screen === 'gallery') { renderer.render(scene, camera); requestAnimationFrame(frame); return; }
  render(now, dt);
  requestAnimationFrame(frame);
}
function stepGame(now, dt) {
  const m = S.match;
  if (S.intro > 0) { S.intro -= dt; return; }
  if (S.zoneT > 0) S.zoneT -= dt;
  if (S.slowT > 0) { S.slowT -= dt; if (S.slowT <= 0) S.slow = 1; }
  if (hintFlash > 0) hintFlash -= dt;
  const mv = moveVector(); m.setMove(mv.x, mv.z);
  if (S.freeze > 0) S.freeze -= dt;
  else m.step(dt * LEVELS[S.levelKey].time * (S.zoneT > 0 ? 0.5 : 1) * S.slow);
  handleEvents();
  if (S.nextAt && now >= S.nextAt) { S.nextAt = null; m.nextRally(); trailPts = []; S.oppHitReal = null; }
  if (S.resultAt && now >= S.resultAt) { S.resultAt = null; showResult(); }
  if (m.phase === 'serve' && m.server === 'P' && S.oppHitReal == null) S.oppHitReal = performance.now();
}

const tmpV = new THREE.Vector3(), UP = new THREE.Vector3(0, 1, 0);
function render(now, dt) {
  const m = cur();
  if (m && humans.P) {
    animateHuman(humans.P, 'P', m, dt);
    animateHuman(humans.O, 'O', m, dt);
    const sp = m.phase === 'serve' ? m.servePoint() : m.shuttle ? m.shuttle.p : null;
    if (sp) {
      shuttle.visible = true; shuttle.position.set(sp.x, sp.y, sp.z);
      const v = m.phase === 'serve' ? { x: 0, y: 1, z: 0 } : m.shuttle.v;
      tmpV.set(v.x, v.y, v.z); if (tmpV.lengthSq() > 1e-6) shuttle.quaternion.setFromUnitVectors(UP, tmpV.normalize());
      if (m.shuttle && m.shuttle.tumble > 0) shuttle.rotateX(now / 1000 * 30);
      shuttleShadow.position.set(sp.x, 0.011, sp.z); shuttleShadow.material.opacity = Math.max(0.15, 0.45 - sp.y * 0.03);
      const lp = dropLine.geometry.attributes.position.array;
      lp[0] = sp.x; lp[1] = sp.y; lp[2] = sp.z; lp[3] = sp.x; lp[4] = 0.01; lp[5] = sp.z; dropLine.geometry.attributes.position.needsUpdate = true;
      if (m.phase === 'rally') { trailPts.push([sp.x, sp.y, sp.z]); if (trailPts.length > TRAIL_N) trailPts.shift(); }
      const arr = trailGeo.attributes.position.array;
      for (let i = 0; i < TRAIL_N; i++) { const q = trailPts[Math.max(0, trailPts.length - TRAIL_N + i)] || [sp.x, sp.y, sp.z]; arr.set(q, i * 3); }
      trailGeo.attributes.position.needsUpdate = true;
    }
    if (m.phase === 'serve') {
      const rs = m.server === 'P' ? 1 : -1;
      serveGlow.visible = true; serveGlow.scale.set(C.L - C.SHORT, C.W, 1);
      serveGlow.position.set(rs * (C.SHORT + C.L) / 2, 0.008, m.serveInfo.zSign * C.W / 2);
    } else serveGlow.visible = false;
  }
  updateAim();
  updateSparks(dt);
  updateCamera(dt);
  updateHint();
  renderer.render(scene, camera);
}

let pvTick = 0;
function updateAim() {
  const m = S.match;
  const active = S.screen === 'play' && m && S.aim && (m.canCommit() || (m.phase === 'rally' && m.lastHitter === 'O'));
  if (!active) { guide.visible = false; reticle.visible = false; $('aim').hidden = true; return; }
  reticle.visible = true; reticle.position.set(S.aim.x, 0.014, S.aim.z);
  // プレビューは狙いか条件が変わったとき（と数フレームおき）に計算
  const keyStr = `${S.aim.x.toFixed(2)},${S.aim.z.toFixed(2)},${S.hclass},${S.face},${m.phase}`;
  if (keyStr !== S.aimKey || ++pvTick % 8 === 0) { S.aimKey = keyStr; S.pv = m.preview(intentNow()); }
  const pv = S.pv; if (!pv) return;
  const pts = pv.sim.path, frac = GUIDE[S.levelKey];
  const n = Math.max(2, Math.min(GUIDE_N, Math.floor(pts.length * frac)));
  const arr = guideGeo.attributes.position.array;
  for (let i = 0; i < GUIDE_N; i++) { const q = pts[Math.min(i, n - 1)]; arr[i * 3] = q.x; arr[i * 3 + 1] = q.y; arr[i * 3 + 2] = q.z; }
  guideGeo.attributes.position.needsUpdate = true; guideGeo.setDrawRange(0, n); guide.computeLineDistances();
  const serve = m.phase === 'serve';
  const ok = pv.sim.end === 'land' && pv.sim.p.x > 0 && pv.sim.p.x <= C.L && Math.abs(pv.sim.p.z) <= C.W &&
    (!serve || (pv.sim.p.x >= C.SHORT && Math.sign(pv.sim.p.z) === m.serveInfo.zSign));
  const col = ok ? COL.you : COL.cpu;
  guideMat.color.setHex(col); reticle.userData.mat.color.setHex(col);
  guide.visible = true;
  $('aim').hidden = false;
  $('aName').textContent = pv.name;
  $('aZone').textContent = zoneName(S.aim);
  $('aSpeed').textContent = Math.round(pv.res.speed * 3.6);
  $('aElev').textContent = `${pv.res.elev >= 0 ? '↗' : '↘'}${Math.abs(pv.res.elev).toFixed(0)}°`;
  let warn = '';
  if (pv.sim.end === 'net') warn = 'ネットにかかる';
  else if (!ok) warn = serve ? '対角のサービスコートの外' : 'アウト';
  else if (pv.sol.downgraded) warn = `この打点からは「${H_NAMES[pv.sol.downgraded]}」は無理 → ${H_NAMES[pv.sol.hclass]}`;
  else if (pv.sim.clearance != null && pv.sim.clearance < 0.15) warn = '白帯すれすれ';
  $('aWarn').textContent = warn;
}

function updateSparks(dt) {
  for (const s of sparks) {
    if (s.life <= 0) continue;
    s.life -= dt; s.v.y -= 9.8 * dt * 0.5; s.m.position.addScaledVector(s.v, dt);
    s.m.material.opacity = Math.max(0, s.life * 2.5); if (s.life <= 0) s.m.visible = false;
  }
  shockT += dt; const k = shockT / 0.35; shock.visible = k < 1;
  if (k < 1) { shock.scale.setScalar(1 + k * 9); shock.material.opacity = 0.7 * (1 - k); }
}

// 自コートのベースライン後方から、ネット越しに相手コートを見る
const camPos = new THREE.Vector3(-8.8, 7, 0), camLook = new THREE.Vector3(2.5, 0, 0);
let updateCamera = function (dt) {
  const m = cur();
  const p = m ? m.pl.P : { x: -3, z: 0 };
  const portrait = camera.aspect < 1;
  // 高めから見下ろすと、相手コートが画面の縦方向に大きく映って狙いやすい
  const back = portrait ? 10.6 : 8.8;
  const want = new THREE.Vector3(-back + (p.x + 3) * 0.2, portrait ? 8.4 : 7.0, p.z * 0.3);
  const look = new THREE.Vector3(portrait ? 1.6 : 2.5, 0, p.z * 0.12);
  const k = Math.min(1, dt * 3);
  camPos.lerp(want, k); camLook.lerp(look, k);
  let sx = 0, sy = 0;
  if (S.shake > 0) { S.shake -= dt; sx = (Math.random() - 0.5) * S.shake; sy = (Math.random() - 0.5) * S.shake; }
  camera.position.set(camPos.x, camPos.y + sy, camPos.z + sx);
  camera.lookAt(camLook);
};

function updateHint() {
  if (hintFlash > 0 || S.screen !== 'play' || !S.match) return;
  const m = S.match, h = $('hint');
  let t = '';
  if (m.phase === 'serve') t = m.server === 'P' ? (fine ? '光っている対角のサービスコートを狙ってクリック' : '光っている対角のサービスコートを指でなぞって離す') : '相手のサーブ';
  else if (m.phase === 'rally' && m.lastHitter === 'O') t = m.commit ? '予約済み。打点に入ったら自動で打つ' : (fine ? '相手コートを狙ってクリック（1〜4 で高さ、Q/E で面）' : '相手コートを指でなぞって、離すと予約');
  h.textContent = t;
}

// =====================================================================
// 結果と育成
// =====================================================================
function showResult() {
  const m = S.match;
  S.screen = 'result'; $('touch').hidden = true; S.aim = null;
  const mine = m.stats.hits.filter(h => h.by === 'P');
  const won = m.score.P > m.score.O;
  $('resTitle').textContent = won ? 'WIN' : 'LOSE';
  $('resTitle').style.color = won ? 'var(--you)' : 'var(--cpu)';
  $('resScore').textContent = `${S.chP.name} ${m.score.P} – ${m.score.O} ${S.chO.name} · ${LEVELS[S.levelKey].jp}`;
  const smashes = mine.filter(h => h.name.includes('スマッシュ'));
  $('rsSmash').textContent = smashes.length ? Math.round(Math.max(...smashes.map(h => h.kmh))) + 'km/h' : '—';
  const aces = m.stats.points.filter(p => p.winner === 'P' && p.reason === 'in').length;
  const errs = m.stats.points.filter(p => p.by === 'P' && p.reason !== 'in').length;
  $('rsAce').textContent = aces; $('rsErr').textContent = errs;
  const avg = a => a.reduce((x, y) => x + y, 0) / a.length;
  $('rsTime').textContent = S.decisions.length ? avg(S.decisions).toFixed(2) + 's' : '—';
  const margins = mine.filter(h => !h.serve && h.landIn && isFinite(h.oppMargin)).map(h => Math.max(0, h.oppMargin));
  $('rsMargin').textContent = margins.length ? avg(margins).toFixed(2) + 's' : '—';
  const late = mine.filter(h => h.late).length; $('rsLate').textContent = late;

  const tips = [];
  const netSmash = m.stats.points.filter(p => p.by === 'P' && p.reason === 'net' && (p.shot || '').includes('スマッシュ')).length;
  const read = mine.filter(h => h.commitLead > 0.6).length;
  const down = mine.filter(h => h.downgraded).length;
  const sliced = mine.filter(h => Math.abs(h.slice) > 0.2).length;
  if (netSmash >= 2) tips.push(['スマッシュがネットに', `${netSmash}本。奥からは狙いを深めに。ジャンプで打点を上げると角度が付く。`]);
  if (down >= 3) tips.push(['打点が低い', `${down}回「沈める」が間に合わなかった。低い打点では「高く」か「ふつう」で立て直す。`]);
  if (late >= 3) tips.push(['判断遅れ', `${late}回。シャトルが頂点を越える前に狙いを決める。`]);
  if (read >= 4) tips.push(['読まれている', `早すぎる予約が${read}回。届く直前まで溜めると相手の一歩目が遅れる。`]);
  if (margins.length && avg(margins) > 0.4) tips.push(['相手に余裕', `平均 ${avg(margins).toFixed(2)}秒。前後の対角（コーナー）へ揺さぶる。`]);
  if (mine.length > 10 && sliced === 0) tips.push(['面を使っていない', 'カットは球が遅く短く曲がり、相手の一歩目を遅らせる。']);
  if (!tips.length) tips.push(['いい試合', '次は CPU を強くするか、マニュアル移動で。']);
  const adv = $('rsAdvice'); adv.innerHTML = '';
  tips.slice(0, 2).forEach(([b, t], i) => { if (i) adv.append(' '); const e = document.createElement('b'); e.textContent = b + '：'; adv.append(e, t); });

  // 経験値
  const hard = mine.filter(h => h.landIn && h.oppMargin < 0.2).length;
  const res = awardMatch(store, S.selP, { won, aces, hardShots: hard, points: m.score.P });
  $('lvUp').textContent = res.levelsUp ? `LEVEL UP!  Lv.${res.ch.level}（能力ポイント +${res.levelsUp * 3}）` : `経験値 +${res.gained}`;
  renderGrowth($('xpBox2'), $('statlist2'), S.selP);
  $('resultScreen').hidden = false;
}

function renderGrowth(xpEl, listEl, id, onChange) {
  const ch = getCharacter(id, store);
  xpEl.innerHTML = '';
  const need = xpToNext(ch.level);
  const line = document.createElement('div');
  line.textContent = `Lv.${ch.level} · 次まで ${need - ch.xp} XP · 能力ポイント ${ch.points} · 戦績 ${ch.record.w}勝${ch.record.l}敗`;
  const bar = document.createElement('div'); bar.className = 'bar'; const fill = document.createElement('span'); fill.style.width = Math.round(ch.xp / need * 100) + '%'; bar.append(fill);
  xpEl.append(line, bar);
  listEl.innerHTML = '';
  for (const k of STAT_KEYS) {
    const row = document.createElement('div'); row.className = 'stat'; row.title = STAT_HINT[k];
    const name = document.createElement('span'); name.textContent = STAT_JP[k];
    const b = document.createElement('div'); b.className = 'bar'; const f = document.createElement('span'); f.style.width = ch.stats[k] + '%'; f.style.background = 'var(--good)'; b.append(f);
    const v = document.createElement('span'); v.className = 'v'; v.textContent = ch.stats[k];
    const plus = document.createElement('button'); plus.type = 'button'; plus.className = 'plus'; plus.textContent = '+';
    plus.setAttribute('aria-label', `${STAT_JP[k]}を上げる（${costFor(ch.stats[k])}ポイント）`);
    plus.disabled = ch.points < costFor(ch.stats[k]) || ch.stats[k] >= 99;
    plus.addEventListener('click', () => { allocate(store, id, k); renderGrowth(xpEl, listEl, id, onChange); if (onChange) onChange(); });
    row.append(name, b, v, plus); listEl.append(row);
  }
}

// =====================================================================
// 選手選択
// =====================================================================
function radarSVG(svg, stats, color) {
  const R = 64, n = STAT_KEYS.length;
  const pt = (i, r) => { const a = -Math.PI / 2 + i * 2 * Math.PI / n; return [Math.cos(a) * r, Math.sin(a) * r]; };
  let html = '';
  for (const f of [0.25, 0.5, 0.75, 1]) html += `<polygon points="${STAT_KEYS.map((_, i) => pt(i, R * f).join(',')).join(' ')}" fill="none" stroke="rgba(142,163,154,0.28)" stroke-width="1"/>`;
  html += `<polygon points="${STAT_KEYS.map((k, i) => pt(i, R * stats[k] / 100).join(',')).join(' ')}" fill="${color}" fill-opacity="0.28" stroke="${color}" stroke-width="2"/>`;
  STAT_KEYS.forEach((k, i) => { const [x, y] = pt(i, R + 16); html += `<text x="${x}" y="${y}" fill="#8ea39a" font-size="11" text-anchor="middle" dominant-baseline="middle">${STAT_JP[k]}</text>`; });
  svg.innerHTML = html;
}
const hex = c => '#' + c.toString(16).padStart(6, '0');
function renderSelect() {
  const roster = $('roster'); roster.innerHTML = '';
  for (const base of CHARACTERS) {
    const ch = getCharacter(base.id, store);
    const b = document.createElement('button'); b.type = 'button'; b.className = 'pc' + (ch.id === S.selP ? ' on' : '') + (ch.id === S.selO ? ' opp' : '');
    const sw = document.createElement('i'); sw.className = 'swatch'; sw.style.background = hex(ch.shirt);
    const nm = document.createElement('strong'); nm.textContent = ch.name;
    const st = document.createElement('span'); st.textContent = `${ch.style} · ${Math.round(ch.height * 100)}cm · ${ch.lefty ? '左' : '右'}利き · Lv.${ch.level}`;
    const ov = document.createElement('span'); ov.className = 'ovr'; ov.textContent = overall(ch);
    b.append(sw, nm, st, ov);
    b.addEventListener('click', () => { S.selP = ch.id; if (S.selO === ch.id) S.selO = null; renderSelect(); newDemo(); });
    b.addEventListener('contextmenu', e => { e.preventDefault(); if (ch.id !== S.selP) { S.selO = ch.id; renderSelect(); newDemo(); } });
    let lp = null;
    b.addEventListener('touchstart', () => { lp = setTimeout(() => { if (ch.id !== S.selP) { S.selO = ch.id; renderSelect(); newDemo(); } }, 550); }, { passive: true });
    b.addEventListener('touchend', () => clearTimeout(lp));
    roster.append(b);
  }
  const ch = getCharacter(S.selP, store);
  $('dName').value = ch.name;
  $('dMeta').innerHTML = '';
  const meta = [['スタイル', ch.style], ['身長', `${Math.round(ch.height * 100)}cm`], ['利き手', ch.lefty ? '左' : '右'], ['総合', overall(ch)]];
  meta.forEach(([k, v], i) => { if (i) $('dMeta').append(' · '); const b = document.createElement('b'); b.textContent = v; $('dMeta').append(`${k} `, b); });
  $('dNote').textContent = ch.note;
  radarSVG($('radar'), ch.stats, hex(ch.shirt === 0xf1f4ec ? COL.you : ch.shirt));
  renderGrowth($('xpBox'), $('statlist'), ch.id, () => radarSVG($('radar'), getCharacter(S.selP, store).stats, hex(ch.shirt === 0xf1f4ec ? COL.you : ch.shirt)));
  $('oppName').textContent = S.selO ? getCharacter(S.selO, store).name : 'おまかせ';
}
$('dName').addEventListener('change', e => { rename(store, S.selP, e.target.value); renderSelect(); });
$('oppRandom').addEventListener('click', () => { S.selO = null; renderSelect(); newDemo(); });
$('startBtn').addEventListener('click', startMatch);
$('againBtn').addEventListener('click', startMatch);
$('menuBtn').addEventListener('click', toTitle);

// 前回の設定
const saved = store.get('arena.settings');
if (saved) {
  const set = id => { const el = $(id); if (el) el.checked = true; };
  set('lv-' + saved.lv); set('mv-' + saved.mv); set('len-' + saved.len);
  if (saved.p && CHARACTERS.some(c => c.id === saved.p)) S.selP = saved.p;
  if (saved.o && CHARACTERS.some(c => c.id === saved.o) && saved.o !== S.selP) S.selO = saved.o;
}
setFace(0); setHeight('high');
resize();
renderSelect();
newDemo();
requestAnimationFrame(frame);

// 動作確認用（URL に ?debug を付けたときだけ）
if (location.search.includes('debug')) {
  window.__arena = S;
  S.project = (x, y, z) => { const v = new THREE.Vector3(x, y, z).project(camera); return [(v.x + 1) / 2 * innerWidth, (1 - v.y) / 2 * innerHeight]; };
}

// 体とフォームの確認用ギャラリー（URL に ?gallery を付けたときだけ）
if (location.search.includes('gallery')) {
  S.screen = 'gallery';
  $('titleScreen').hidden = true;
  disposeHuman(humans.P); disposeHuman(humans.O); humans.P = humans.O = null; demo = null;
  shuttle.visible = false;
  const list = [['ready'], ['prep_over'], ['smash_c'], ['smash_f'], ['drive_c'], ['bh_c'], ['net_fh_c', LUNGE], ['lift_fh_f', LUNGE], ['ss_p']];
  const chs = ['ren', 'jin', 'takeru', 'mio', 'yu', 'rookie', 'ren', 'mio', 'jin'];
  list.forEach(([name, legs], i) => {
    const h = makeHuman(getCharacter(chs[i], store), COL.you);
    const x = -6 + i * 1.5;
    h.root.position.set(0, 0, x); h.ring.position.set(0, 0.012, x); h.shadow.position.set(0, 0.01, x);
    h.root.rotation.y = -Math.PI / 2 + 0.5;
    const pose = legs ? withLegs(poseOf(name), legs) : poseOf(name);
    h.J.hips.position.y = 0.94 + (pose.hipsY ?? -0.07);
    for (const k2 of JOINTS) { const v = pose[k2]; if (v && h.J[k2]) h.J[k2].rotation.set(v[0], v[1], v[2]); }
  });
  camera.position.set(-6.5, 1.6, 0); camera.lookAt(0, 1.0, 0);
  const g = () => { renderer.render(scene, camera); requestAnimationFrame(g); };
  updateCamera = () => {};
}
