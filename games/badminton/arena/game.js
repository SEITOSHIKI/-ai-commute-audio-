// ラリーIQ アリーナ — 3D 描画と操作。試合のルールと物理は match.js / physics.js。
import * as THREE from './vendor/three.module.js';
import { C, LEVELS } from './physics.js';
import { Match } from './match.js';

const $ = id => document.getElementById(id);
const store = {
  get(k) { try { return JSON.parse(localStorage.getItem(k)); } catch (e) { return null; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* 保存できなくても遊べる */ } },
};
const COL = { you: 0xf4b63f, cpu: 0xee6c5a, mat: 0x1f5e4c, floor: 0x132326, line: 0xf1f4ec, skin: 0xe6bf98, dark: 0x1b272b };
const GUIDE = { beginner: 1, club: 0.6, expert: 0.35, pro: 0.18 };
const FACE_NAMES = { '-2': 'リバース強', '-1': 'リバース', '0': 'フラット', '1': 'カット', '2': 'カット強' };
const fine = matchMedia('(pointer: fine)').matches;

// =====================================================================
// シーン
// =====================================================================
const view = $('view');
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
view.appendChild(renderer.domElement);

const ui = document.createElement('canvas'); // ドラッグ矢印とスティックの描画
Object.assign(ui.style, { position: 'fixed', inset: '0', width: '100%', height: '100%', pointerEvents: 'none' });
document.body.insertBefore(ui, $('hud'));
const uictx = ui.getContext('2d');

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b1416);
scene.fog = new THREE.Fog(0x0b1416, 24, 58);
const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 200);

scene.add(new THREE.HemisphereLight(0xe4f0ea, 0x1a2a2c, 1.0));
const sun = new THREE.DirectionalLight(0xffffff, 1.7);
sun.position.set(3, 14, 9);
scene.add(sun);

function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  ui.width = Math.round(w * dpr); ui.height = Math.round(h * dpr);
  uictx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener('resize', resize);

// ---- 床とコート ----
function courtTexture() {
  const S = 100, W = 14.6, H = 7.3;
  const c = document.createElement('canvas'); c.width = W * S; c.height = H * S;
  const g = c.getContext('2d');
  g.fillStyle = '#1f5e4c'; g.fillRect(0, 0, c.width, c.height);
  // うっすらマットの継ぎ目
  g.strokeStyle = 'rgba(0,0,0,0.08)'; g.lineWidth = 2;
  for (let z = -3; z <= 3; z += 1.5) { g.beginPath(); g.moveTo(0, (z + H / 2) * S); g.lineTo(c.width, (z + H / 2) * S); g.stroke(); }
  g.strokeStyle = '#f1f4ec'; g.lineWidth = 4; g.lineCap = 'square';
  const L = (x0, z0, x1, z1) => { g.beginPath(); g.moveTo((x0 + W / 2) * S, (z0 + H / 2) * S); g.lineTo((x1 + W / 2) * S, (z1 + H / 2) * S); g.stroke(); };
  L(-6.7, -3.05, 6.7, -3.05); L(-6.7, 3.05, 6.7, 3.05); L(-6.7, -3.05, -6.7, 3.05); L(6.7, -3.05, 6.7, 3.05);
  L(-6.7, -C.W, 6.7, -C.W); L(-6.7, C.W, 6.7, C.W);
  for (const s of [-1, 1]) {
    L(s * C.SHORT, -3.05, s * C.SHORT, 3.05);
    L(s * 5.94, -3.05, s * 5.94, 3.05);
    L(s * C.SHORT, 0, s * 6.7, 0);
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = renderer.capabilities.getMaxAnisotropy();
  return t;
}
const floor = new THREE.Mesh(new THREE.PlaneGeometry(80, 50), new THREE.MeshStandardMaterial({ color: COL.floor, roughness: 0.85 }));
floor.rotation.x = -Math.PI / 2; scene.add(floor);
const court = new THREE.Mesh(new THREE.PlaneGeometry(14.6, 7.3), new THREE.MeshStandardMaterial({ map: courtTexture(), roughness: 0.7 }));
court.rotation.x = -Math.PI / 2; court.position.y = 0.004; scene.add(court);

// 対角のサービスコートを光らせる（サーブ時）
const serveGlow = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ color: COL.line, transparent: true, opacity: 0.12, depthWrite: false }));
serveGlow.rotation.x = -Math.PI / 2; serveGlow.position.y = 0.008; serveGlow.visible = false; scene.add(serveGlow);

// ---- ネット ----
function netTexture() {
  const c = document.createElement('canvas'); c.width = 512; c.height = 64;
  const g = c.getContext('2d');
  g.strokeStyle = 'rgba(20,24,24,0.9)'; g.lineWidth = 1.2;
  for (let x = 0; x <= 512; x += 6) { g.beginPath(); g.moveTo(x, 0); g.lineTo(x, 64); g.stroke(); }
  for (let y = 0; y <= 64; y += 6) { g.beginPath(); g.moveTo(0, y); g.lineTo(512, y); g.stroke(); }
  const t = new THREE.CanvasTexture(c); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(4, 1);
  return t;
}
const net = new THREE.Mesh(new THREE.PlaneGeometry(6.1, 0.76), new THREE.MeshBasicMaterial({ map: netTexture(), transparent: true, side: THREE.DoubleSide, depthWrite: false }));
net.rotation.y = Math.PI / 2; net.position.set(0, C.NET_H - 0.38, 0); scene.add(net);
const tape = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.05, 6.1), new THREE.MeshStandardMaterial({ color: 0xf6f7f2 }));
tape.position.set(0, C.NET_H - 0.025, 0); scene.add(tape);
for (const z of [-3.12, 3.12]) {
  const post = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.05, 1.55, 12), new THREE.MeshStandardMaterial({ color: 0xc9d2cf, metalness: 0.4, roughness: 0.4 }));
  post.position.set(0, 0.775, z); scene.add(post);
}

// ---- アリーナの背景（照明と観客席の帯） ----
const wall = new THREE.Mesh(new THREE.PlaneGeometry(90, 20), new THREE.MeshBasicMaterial({ color: 0x0f1c1f }));
wall.position.set(0, 8, -16); scene.add(wall);
for (let i = -6; i <= 6; i++) {
  const strip = new THREE.Mesh(new THREE.PlaneGeometry(4.6, 0.12), new THREE.MeshBasicMaterial({ color: i % 3 === 0 ? 0xf4b63f : 0x3a5a5c }));
  strip.position.set(i * 6.2, 3.2, -15.9); scene.add(strip);
}
for (let i = 0; i < 3; i++) for (const s of [-1, 1]) {
  const cone = new THREE.Mesh(new THREE.ConeGeometry(2.6, 14, 24, 1, true), new THREE.MeshBasicMaterial({ color: 0xfaf4e0, transparent: true, opacity: 0.035, depthWrite: false, side: THREE.DoubleSide, blending: THREE.AdditiveBlending }));
  cone.position.set(s * (2.5 + i * 4.2), 7, -2.5 - i * 1.5); scene.add(cone);
}

// ---- 影（ブロブ） ----
const blobMat = new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.32, depthWrite: false });
function blob(r) { const m = new THREE.Mesh(new THREE.CircleGeometry(r, 24), blobMat.clone()); m.rotation.x = -Math.PI / 2; m.position.y = 0.01; scene.add(m); return m; }

// =====================================================================
// 選手
// =====================================================================
function makeAthlete(color) {
  const root = new THREE.Group();
  const body = new THREE.Group(); root.add(body);
  const shirt = new THREE.MeshStandardMaterial({ color, roughness: 0.6 });
  const dark = new THREE.MeshStandardMaterial({ color: COL.dark, roughness: 0.8 });
  const skin = new THREE.MeshStandardMaterial({ color: COL.skin, roughness: 0.7 });
  const white = new THREE.MeshStandardMaterial({ color: 0xf2f4f0, roughness: 0.5 });

  const legs = [];
  for (const z of [-0.1, 0.1]) {
    const hip = new THREE.Group(); hip.position.set(0, 0.86, z);
    const leg = new THREE.Mesh(new THREE.CapsuleGeometry(0.075, 0.62, 4, 10), dark); leg.position.y = -0.4; hip.add(leg);
    const shoe = new THREE.Mesh(new THREE.BoxGeometry(0.26, 0.08, 0.11), white); shoe.position.set(0.05, -0.8, 0); hip.add(shoe);
    body.add(hip); legs.push(hip);
  }
  const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.18, 0.38, 6, 14), shirt); torso.position.y = 1.18; body.add(torso);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.125, 18, 14), skin); head.position.y = 1.62; body.add(head);
  const band = new THREE.Mesh(new THREE.TorusGeometry(0.125, 0.018, 6, 20), shirt); band.position.y = 1.66; band.rotation.x = Math.PI / 2; body.add(band);

  // 利き腕（右）。腕の付け根を回して素振り
  const arm = new THREE.Group(); arm.position.set(0, 1.42, 0.22); body.add(arm);
  const upper = new THREE.Mesh(new THREE.CapsuleGeometry(0.048, 0.5, 4, 8), skin); upper.position.y = -0.3; arm.add(upper);
  const racket = new THREE.Group(); racket.position.y = -0.6; arm.add(racket);
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.016, 0.42, 8), dark); shaft.position.y = -0.21; racket.add(shaft);
  const frame = new THREE.Mesh(new THREE.TorusGeometry(0.11, 0.009, 6, 28), new THREE.MeshStandardMaterial({ color, metalness: 0.3, roughness: 0.4 }));
  frame.scale.set(1, 1.32, 1); frame.position.y = -0.56; frame.rotation.y = Math.PI / 2; racket.add(frame);
  const strings = new THREE.Mesh(new THREE.CircleGeometry(0.105, 20), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.22, side: THREE.DoubleSide }));
  strings.scale.set(1, 1.32, 1); strings.position.y = -0.56; strings.rotation.y = Math.PI / 2; racket.add(strings);
  const off = new THREE.Group(); off.position.set(0, 1.42, -0.22); body.add(off);
  const offArm = new THREE.Mesh(new THREE.CapsuleGeometry(0.045, 0.48, 4, 8), skin); offArm.position.y = -0.28; off.add(offArm);

  scene.add(root);
  return { root, body, legs, arm, off, racket, phase: 0, armAngle: 0.6, shadow: blob(0.36) };
}
const athletes = { P: makeAthlete(COL.you), O: makeAthlete(COL.cpu) };
athletes.O.body.rotation.y = Math.PI;

// =====================================================================
// シャトル・弾道ガイド・エフェクト
// =====================================================================
const shuttle = new THREE.Group();
{
  const cork = new THREE.Mesh(new THREE.SphereGeometry(0.034, 14, 10, 0, Math.PI * 2, 0, Math.PI / 2), new THREE.MeshStandardMaterial({ color: 0xf4b63f, roughness: 0.5 }));
  cork.rotation.x = 0; cork.position.y = 0.0;
  const skirt = new THREE.Mesh(new THREE.ConeGeometry(0.05, 0.09, 14, 1, true), new THREE.MeshStandardMaterial({ color: 0xffffff, side: THREE.DoubleSide, transparent: true, opacity: 0.95 }));
  skirt.position.y = -0.045; skirt.rotation.x = Math.PI;
  shuttle.add(cork, skirt);
  shuttle.scale.setScalar(2.6); // 真横の引きの画でも見えるよう大きめ
  scene.add(shuttle);
}
const shuttleShadow = blob(0.12);
const dropLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]), new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.18 }));
scene.add(dropLine);
const TRAIL_N = 26;
const trailGeo = new THREE.BufferGeometry();
trailGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(TRAIL_N * 3), 3));
const trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.5 }));
scene.add(trail);
let trailPts = [];

const GUIDE_N = 400;
const guideGeo = new THREE.BufferGeometry();
guideGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(GUIDE_N * 3), 3));
const guideMat = new THREE.LineDashedMaterial({ color: COL.you, dashSize: 0.18, gapSize: 0.12, transparent: true, opacity: 0.9 });
const guide = new THREE.Line(guideGeo, guideMat); guide.visible = false; scene.add(guide);
const landRing = new THREE.Mesh(new THREE.RingGeometry(0.16, 0.24, 28), new THREE.MeshBasicMaterial({ color: COL.you, transparent: true, opacity: 0.9, depthWrite: false }));
landRing.rotation.x = -Math.PI / 2; landRing.position.y = 0.012; landRing.visible = false; scene.add(landRing);

// ヒットのしぶきと衝撃波
const sparks = [];
const sparkGeo = new THREE.SphereGeometry(0.03, 6, 4);
for (let i = 0; i < 40; i++) {
  const m = new THREE.Mesh(sparkGeo, new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true }));
  m.visible = false; scene.add(m); sparks.push({ m, v: new THREE.Vector3(), life: 0 });
}
const shock = new THREE.Mesh(new THREE.RingGeometry(0.2, 0.28, 40), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false }));
scene.add(shock);
let shockT = 9;
function burst(p, color, n, speed) {
  let k = 0;
  for (const s of sparks) {
    if (s.life > 0) continue;
    s.m.visible = true; s.m.position.set(p.x, p.y, p.z); s.m.material.color.setHex(color);
    s.v.set((Math.random() - 0.5), Math.random() * 0.8, (Math.random() - 0.5)).normalize().multiplyScalar(speed * (0.5 + Math.random()));
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
// 状態と入力
// =====================================================================
const S = {
  screen: 'title', match: null, levelKey: 'club', autoMove: true, target: 11,
  face: 0, zone: 0, zoneT: 0, slow: 1, slowT: 0, freeze: 0, shake: 0,
  drag: null, stick: null, keys: new Set(), yaw: 0,
  oppHitReal: null, decisions: [], banner: null, nextAt: null, resultAt: null, intro: 0,
  cam: { x: 0, dist: 18 },
};

// タイトル画面の背景で流す CPU 同士のデモ
let demo = new Match({ human: false, level: 'expert', levelP: 'expert' });
function demoTick(dt) {
  demo.step(dt * 0.55);
  demo.events.length = 0;
  if (demo.phase === 'dead' && demo.phaseT > 0.6) demo.nextRally();
  if (demo.phase === 'over') demo = new Match({ human: false, level: 'expert', levelP: 'expert' });
}
const cur = () => (S.screen === 'title' ? demo : S.match);

function aimFromDrag(d) {
  const dx = d.x - d.x0, dy = d.y - d.y0;
  const len = Math.hypot(dx, dy);
  const R = Math.max(90, Math.min(220, Math.min(innerWidth, innerHeight) * 0.3));
  const elev = Math.atan2(-dy, Math.max(Math.abs(dx), 1)) * 180 / Math.PI;
  const power = Math.pow(Math.min(1, len / R), 1.6);
  return { elev: Math.max(-40, Math.min(75, elev)), yaw: currentYaw(), power: Math.max(0.01, power), slice: S.face * 0.5, len, R };
}
function currentYaw() {
  let y = 0;
  if (S.keys.has('w') || S.keys.has('arrowup')) y -= 1;
  if (S.keys.has('s') || S.keys.has('arrowdown')) y += 1;
  if (S.stick) y += S.stick.vy;
  return Math.max(-1, Math.min(1, y));
}
function moveVector() {
  let x = 0, z = 0;
  if (S.keys.has('a') || S.keys.has('arrowleft')) x -= 1;
  if (S.keys.has('d') || S.keys.has('arrowright')) x += 1;
  if (S.keys.has('w') || S.keys.has('arrowup')) z -= 1;
  if (S.keys.has('s') || S.keys.has('arrowdown')) z += 1;
  if (S.stick) { x += S.stick.vx; z += S.stick.vy; }
  return { x, z };
}

const canvas = renderer.domElement;
canvas.addEventListener('pointerdown', e => {
  if (S.screen !== 'play') return;
  initAudio();
  const stickEl = $('stick').getBoundingClientRect();
  const inStick = e.pointerType !== 'mouse' && e.clientX < stickEl.right + 40 && e.clientY > stickEl.top - 40;
  if (inStick && !S.stick) {
    const cx = stickEl.left + stickEl.width / 2, cy = stickEl.top + stickEl.height / 2;
    S.stick = { id: e.pointerId, cx, cy, vx: 0, vy: 0 };
    updateStick(e);
  } else if (!S.drag) {
    S.drag = { id: e.pointerId, x0: e.clientX, y0: e.clientY, x: e.clientX, y: e.clientY };
  }
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener('pointermove', e => {
  if (S.stick && e.pointerId === S.stick.id) updateStick(e);
  if (S.drag && e.pointerId === S.drag.id) { S.drag.x = e.clientX; S.drag.y = e.clientY; }
});
const endPointer = e => {
  if (S.stick && e.pointerId === S.stick.id) { S.stick = null; $('knob').style.transform = ''; }
  if (S.drag && e.pointerId === S.drag.id) {
    const d = S.drag; S.drag = null;
    if (e.type === 'pointerup') release(d);
  }
};
canvas.addEventListener('pointerup', endPointer);
canvas.addEventListener('pointercancel', endPointer);
canvas.addEventListener('wheel', e => { if (S.screen === 'play') { e.preventDefault(); setFace(S.face + (e.deltaY > 0 ? 1 : -1)); } }, { passive: false });

function updateStick(e) {
  const st = S.stick, R = 52;
  let dx = e.clientX - st.cx, dy = e.clientY - st.cy;
  const l = Math.hypot(dx, dy); if (l > R) { dx *= R / l; dy *= R / l; }
  st.vx = dx / R; st.vy = dy / R;
  $('knob').style.transform = `translate(${dx}px, ${dy}px)`;
}

function release(d) {
  const m = S.match; if (!m) return;
  let aim = aimFromDrag(d);
  if (aim.len < 12) aim = { elev: 40, yaw: currentYaw(), power: 0.55, slice: S.face * 0.5 }; // タップは無難なロブ
  if (!m.canCommit()) { flashHint(m.phase === 'serve' ? '相手のサーブを待つ' : '相手が打ってから決める'); return; }
  m.commitShot({ elev: aim.elev, yaw: aim.yaw, power: aim.power, slice: aim.slice });
  if (S.oppHitReal != null) S.decisions.push((performance.now() - S.oppHitReal) / 1000);
  S.oppHitReal = null;
  tone(1250, 0.05, 0.05);
}

window.addEventListener('keydown', e => {
  const k = e.key.toLowerCase();
  if (S.screen !== 'play') { if (k === 'enter') startMatch(); return; }
  if ([' ', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright'].includes(k)) e.preventDefault();
  S.keys.add(k);
  if (k === 'q') setFace(S.face - 1);
  if (k === 'e') setFace(S.face + 1);
  if (k === ' ') S.match && S.match.jump('P');
  if (k === 'shift') useZone();
  if (k === 'escape') toTitle();
});
window.addEventListener('keyup', e => S.keys.delete(e.key.toLowerCase()));
window.addEventListener('blur', () => S.keys.clear());

function setFace(f) {
  S.face = Math.max(-2, Math.min(2, f));
  $('faceName').textContent = FACE_NAMES[S.face];
  [...$('notches').children].forEach((n, i) => n.classList.toggle('on', i === S.face + 2));
}
function useZone() {
  if (S.zone < 1 || S.zoneT > 0) return;
  S.zone = 0; S.zoneT = 3; tone(520, 0.4, 0.08, 'sine'); updateZone();
}
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
// 試合の開始・進行
// =====================================================================
function startMatch() {
  S.levelKey = document.querySelector('input[name="lv"]:checked').value;
  S.autoMove = document.querySelector('input[name="mv"]:checked').value === 'auto';
  S.target = +document.querySelector('input[name="len"]:checked').value;
  store.set('arena.settings', { lv: S.levelKey, mv: S.autoMove ? 'auto' : 'manual', len: S.target });
  S.match = new Match({ level: S.levelKey, autoMove: S.autoMove, target: S.target });
  S.decisions = []; S.zone = 0; S.zoneT = 0; S.face = 0; setFace(0); updateZone();
  S.screen = 'play'; S.nextAt = null; S.resultAt = null; S.oppHitReal = null;
  $('titleScreen').hidden = true; $('resultScreen').hidden = true;
  $('touch').hidden = fine;
  $('stickLabel').textContent = S.autoMove ? 'コース' : '移動・コース';
  $('shotcard').hidden = true;
  initAudio();
  updateScore();
  showBanner('READY', '', COL.you, 0.9);
  S.intro = 1.3;
  setTimeout(() => { if (S.screen === 'play') showBanner('GO!', '', COL.you, 0.6); }, 900);
}

function toTitle() {
  S.screen = 'title'; S.match = null; S.drag = null;
  $('resultScreen').hidden = true; $('titleScreen').hidden = false; $('touch').hidden = true;
  hideBanner();
}

let bannerTimer = null;
function showBanner(text, sub, color, dur) {
  const b = $('banner');
  b.hidden = false; b.innerHTML = '';
  b.style.color = '#' + color.toString(16).padStart(6, '0');
  b.append(document.createTextNode(text));
  if (sub) { const s = document.createElement('small'); s.textContent = sub; b.append(s); }
  b.style.animation = 'none'; void b.offsetWidth; b.style.animation = '';
  clearTimeout(bannerTimer);
  bannerTimer = setTimeout(hideBanner, dur * 1000);
}
function hideBanner() { $('banner').hidden = true; }

let hintFlash = 0;
function flashHint(t) { $('hint').textContent = t; hintFlash = 1.2; }

function updateScore() {
  const m = S.match;
  $('ptsP').textContent = m.score.P; $('ptsO').textContent = m.score.O;
  $('srvP').classList.toggle('on', m.server === 'P');
  $('srvO').classList.toggle('on', m.server === 'O');
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
  return { cls: 'r-ok', text: hi > -Infinity && hi >= 0 ? `普通：相手の余裕 ${m.toFixed(2)}秒` : '低く沈めた：相手は上げるしかない' };
}

function handleEvents() {
  const m = S.match;
  for (const ev of m.events.splice(0)) {
    if (ev.type === 'hit') {
      const p = m.shuttle.p;
      const power = Math.min(1, ev.kmh / 300);
      noiseHit(power);
      burst(p, ev.by === 'P' ? COL.you : COL.cpu, 6 + Math.round(power * 14), 1.5 + power * 4);
      if (ev.kmh > 230) {
        S.shake = 0.12 + power * 0.18; S.freeze = 0.05; S.slow = 0.45; S.slowT = 0.25;
        shock.position.set(p.x, p.y, p.z); shock.lookAt(camera.position); shockT = 0;
      }
      if (ev.by === 'P') {
        const r = rate(ev);
        $('shotcard').hidden = false;
        $('scName').textContent = ev.name;
        $('scName').style.color = '#f4b63f';
        const sl = ev.slice ? ` · ${FACE_NAMES[Math.round(ev.slice / 0.5)]}` : '';
        $('scNums').textContent = `${Math.round(ev.kmh)}km/h · ${ev.elev >= 0 ? '↗' : '↘'}${Math.abs(ev.elev).toFixed(0)}°${sl}`;
        const extra = ev.late ? ' · 判断遅れ' : ev.commitLead > 0.6 ? ' · 早く決めすぎて読まれた' : '';
        $('scEval').className = 'eval ' + r.cls; $('scEval').textContent = r.text + extra;
        S.zone = Math.min(1, S.zone + (r.cls === 'r-ace' ? 0.34 : r.cls === 'r-hard' ? 0.2 : r.cls === 'r-ok' ? 0.08 : 0));
        updateZone();
        S.oppHitReal = null;
      } else {
        S.oppHitReal = performance.now();
      }
      trailPts = [];
    } else if (ev.type === 'point') {
      updateScore();
      const youWon = ev.winner === 'P';
      const byYou = ev.by === 'P';
      let head, sub;
      if (ev.reason === 'in') { head = youWon ? 'POINT!' : 'LOST'; sub = `${byYou ? 'あなた' : 'CPU'}の${ev.shot || 'ショット'}${ev.kmh > 200 ? ` ${Math.round(ev.kmh)}km/h` : ''}が決まった`; }
      else if (ev.reason === 'out') { head = 'OUT'; sub = `${byYou ? 'あなた' : 'CPU'}の${ev.shot || 'ショット'}がアウト`; }
      else { head = 'NET'; sub = `${byYou ? 'あなた' : 'CPU'}の${ev.shot || 'ショット'}がネット`; }
      showBanner(head, sub, youWon ? COL.you : COL.cpu, 1.5);
      tone(youWon ? 880 : 300, 0.3, 0.1);
      S.slow = 0.35; S.slowT = 0.7;
      S.nextAt = performance.now() + 1700;
    } else if (ev.type === 'game') {
      S.resultAt = performance.now() + 1900;
      S.nextAt = null;
      setTimeout(() => showBanner('GAME!', `${ev.score.P} – ${ev.score.O}`, ev.winner === 'P' ? COL.you : COL.cpu, 1.8), 700);
    } else if (ev.type === 'serve-ready') {
      updateScore();
    }
  }
}

// =====================================================================
// メインループ
// =====================================================================
let last = performance.now();
function frame(now) {
  const dt = Math.min(0.05, (now - last) / 1000); last = now;
  if (S.screen === 'play' && S.match) stepGame(now, dt);
  else if (S.screen === 'title') demoTick(dt);
  render(now, dt);
  requestAnimationFrame(frame);
}

function stepGame(now, dt) {
  const m = S.match;
  if (S.intro > 0) { S.intro -= dt; return; }
  if (S.zoneT > 0) S.zoneT -= dt;
  if (S.slowT > 0) { S.slowT -= dt; if (S.slowT <= 0) S.slow = 1; }
  if (hintFlash > 0) hintFlash -= dt;
  const mv = moveVector();
  m.setMove(mv.x, mv.z);
  if (S.freeze > 0) { S.freeze -= dt; }
  else {
    const scale = LEVELS[S.levelKey].time * (S.zoneT > 0 ? 0.5 : 1) * S.slow;
    m.step(dt * scale);
  }
  handleEvents();
  if (S.nextAt && now >= S.nextAt) { S.nextAt = null; m.nextRally(); trailPts = []; S.oppHitReal = null; }
  if (S.resultAt && now >= S.resultAt) { S.resultAt = null; showResult(); }
  if (m.phase === 'serve' && m.server === 'P' && S.oppHitReal == null) S.oppHitReal = performance.now();
}

const tmpV = new THREE.Vector3(), UP = new THREE.Vector3(0, 1, 0);
function render(now, dt) {
  const m = cur();
  const t = now / 1000;
  if (m) {
    for (const id of ['P', 'O']) poseAthlete(id, m, dt, t);
    // シャトル
    let sp;
    if (m.phase === 'serve') sp = m.servePoint();
    else sp = m.shuttle ? m.shuttle.p : null;
    if (sp) {
      shuttle.visible = true;
      shuttle.position.set(sp.x, sp.y, sp.z);
      const v = m.phase === 'serve' ? { x: 0, y: 1, z: 0 } : m.shuttle.v;
      tmpV.set(v.x, v.y, v.z);
      if (tmpV.lengthSq() > 1e-6) shuttle.quaternion.setFromUnitVectors(UP, tmpV.normalize());
      if (m.shuttle && m.shuttle.tumble > 0) shuttle.rotateX(t * 30);
      shuttleShadow.position.set(sp.x, 0.011, sp.z);
      shuttleShadow.scale.setScalar(1 + sp.y * 0.08);
      shuttleShadow.material.opacity = Math.max(0.15, 0.45 - sp.y * 0.03);
      const lp = dropLine.geometry.attributes.position.array;
      lp[0] = sp.x; lp[1] = sp.y; lp[2] = sp.z; lp[3] = sp.x; lp[4] = 0.01; lp[5] = sp.z;
      dropLine.geometry.attributes.position.needsUpdate = true;
      if (m.phase === 'rally') { trailPts.push([sp.x, sp.y, sp.z]); if (trailPts.length > TRAIL_N) trailPts.shift(); }
      const arr = trailGeo.attributes.position.array;
      for (let i = 0; i < TRAIL_N; i++) {
        const q = trailPts[Math.max(0, trailPts.length - TRAIL_N + i)] || trailPts[0] || [sp.x, sp.y, sp.z];
        arr[i * 3] = q[0]; arr[i * 3 + 1] = q[1]; arr[i * 3 + 2] = q[2];
      }
      trailGeo.attributes.position.needsUpdate = true;
    }
    // サーブ時は対角のサービスコートを示す
    if (m.phase === 'serve') {
      const rs = m.server === 'P' ? 1 : -1;
      serveGlow.visible = true;
      serveGlow.scale.set(C.L - C.SHORT, C.W, 1);
      serveGlow.position.set(rs * (C.SHORT + C.L) / 2, 0.008, m.serveInfo.zSign * C.W / 2);
    } else serveGlow.visible = false;
  }
  updateGuide();
  updateSparks(dt);
  updateCamera(dt);
  drawUI();
  updateHint();
  renderer.render(scene, camera);
}

function poseAthlete(id, m, dt, t) {
  const a = athletes[id], p = m.pl[id];
  a.root.position.set(p.x, p.y, p.z);
  a.shadow.position.set(p.x, 0.01, p.z);
  a.shadow.scale.setScalar(1 - Math.min(0.4, p.y));
  // 走り
  const sp = Math.hypot(p.vx, p.vz);
  a.phase += dt * (4 + sp * 3.2);
  const k = Math.min(1, sp / 3);
  a.legs[0].rotation.z = Math.sin(a.phase) * 0.7 * k;
  a.legs[1].rotation.z = -Math.sin(a.phase) * 0.7 * k;
  a.body.position.y = Math.abs(Math.sin(a.phase)) * 0.05 * k;
  // 腕: 素振り中 → 予約済みの構え → 待機
  const since = m.time - p.swingT;
  const sh = m.shuttle ? m.shuttle.p : null;
  const overhead = sh ? sh.y > 1.9 : false;
  let target;
  if (since >= 0 && since < 0.22) {
    const u = since / 0.22;
    target = overhead ? 3.9 - u * 3.0 : -0.9 + u * 2.6;
    a.armAngle = target;
  } else {
    const incoming = m.lastHitter && m.lastHitter !== id && m.phase === 'rally';
    const ready = id === 'P' ? !!m.commit : incoming;
    target = ready ? (overhead ? 3.7 : -0.7) : incoming ? 1.2 : 0.5;
    a.armAngle += (target - a.armAngle) * Math.min(1, dt * 10);
  }
  a.arm.rotation.z = a.armAngle;
  a.off.rotation.z = overhead && since > 0.3 ? 2.2 : 0.3;
}

let guideAim = null;
function updateGuide() {
  const m = S.match;
  const show = S.screen === 'play' && m && S.drag && (m.canCommit() || m.phase === 'rally');
  if (!show) { guide.visible = false; landRing.visible = false; $('aim').hidden = true; guideAim = null; return; }
  const aim = aimFromDrag(S.drag);
  if (aim.len < 12) { guide.visible = false; landRing.visible = false; $('aim').hidden = true; return; }
  guideAim = aim;
  const pv = m.preview(aim);
  const pts = pv.sim.path;
  const frac = GUIDE[S.levelKey];
  const n = Math.max(2, Math.min(GUIDE_N, Math.floor(pts.length * frac)));
  const arr = guideGeo.attributes.position.array;
  for (let i = 0; i < GUIDE_N; i++) {
    const q = pts[Math.min(i, n - 1)];
    arr[i * 3] = q.x; arr[i * 3 + 1] = q.y; arr[i * 3 + 2] = q.z;
  }
  guideGeo.attributes.position.needsUpdate = true;
  guideGeo.setDrawRange(0, n);
  guide.computeLineDistances();
  const ok = pv.sim.end === 'land' && Math.abs(pv.sim.p.z) <= C.W && pv.sim.p.x > 0 && pv.sim.p.x <= C.L &&
    (m.phase !== 'serve' || (pv.sim.p.x >= C.SHORT && Math.sign(pv.sim.p.z) === m.serveInfo.zSign));
  guideMat.color.setHex(ok ? COL.you : COL.cpu);
  guide.visible = true;
  landRing.visible = frac >= 0.6 && pv.sim.end === 'land';
  landRing.position.set(pv.sim.p.x, 0.012, pv.sim.p.z);
  landRing.material.color.setHex(ok ? COL.you : COL.cpu);
  // 数値
  $('aim').hidden = false;
  $('aElev').textContent = `${pv.res.elev >= 0 ? '↗' : '↘'}${Math.abs(pv.res.elev).toFixed(0)}°`;
  $('aSpeed').textContent = Math.round(pv.res.speed * 3.6);
  $('aYaw').textContent = aim.yaw < -0.3 ? '奥側へ' : aim.yaw > 0.3 ? '手前側へ' : 'まっすぐ';
  let land = '—';
  if (pv.sim.end === 'net') land = 'ネット';
  else if (!ok) land = 'アウト';
  else {
    const x = pv.sim.p.x;
    land = (x < 2.6 ? '前' : x < 4.6 ? '中' : '奥') + (pv.sim.p.z < -0.8 ? '・奥側' : pv.sim.p.z > 0.8 ? '・手前側' : '');
  }
  if (frac < 0.6) land = pv.sim.end === 'net' ? 'ネット' : '？'; // 上級者は着地を自分で読む
  $('aLand').textContent = land;
}

function updateSparks(dt) {
  for (const s of sparks) {
    if (s.life <= 0) continue;
    s.life -= dt;
    s.v.y -= 9.8 * dt * 0.5;
    s.m.position.addScaledVector(s.v, dt);
    s.m.material.opacity = Math.max(0, s.life * 2.5);
    if (s.life <= 0) s.m.visible = false;
  }
  shockT += dt;
  const k = shockT / 0.35;
  shock.visible = k < 1;
  if (k < 1) { shock.scale.setScalar(1 + k * 9); shock.material.opacity = 0.7 * (1 - k); }
}

function updateCamera(dt) {
  const m = cur();
  let fx = 0, spread = 13;
  if (m) {
    const sp = m.phase === 'serve' ? m.servePoint() : m.shuttle ? m.shuttle.p : { x: 0 };
    const px = m.pl.P.x, ox = m.pl.O.x;
    fx = Math.max(-3.2, Math.min(3.2, sp.x * 0.45 + (px + ox) * 0.25));
    spread = Math.max(Math.abs(px - ox), Math.abs(sp.x - px) + 1, Math.abs(sp.x - ox) + 1) + 3.2;
  }
  // 縦画面では画角を広げ、横幅が収まる距離まで引く
  const fov = camera.aspect >= 1.2 ? 30 : Math.min(64, 30 * 1.2 / camera.aspect);
  if (Math.abs(camera.fov - fov) > 0.01) { camera.fov = fov; camera.updateProjectionMatrix(); }
  const vfov = camera.fov * Math.PI / 180;
  const hfov = 2 * Math.atan(Math.tan(vfov / 2) * camera.aspect);
  const half = camera.aspect >= 1 ? Math.max(6, Math.min(9.2, spread / 2 + 1.2)) : Math.max(4.4, Math.min(7, spread / 2 + 0.4));
  const want = Math.min(34, Math.max(12, half / Math.tan(hfov / 2)));
  S.cam.x += (fx - S.cam.x) * Math.min(1, dt * 2.4);
  S.cam.dist += (want - S.cam.dist) * Math.min(1, dt * 1.6);
  const d = S.cam.dist;
  let sx = 0, sy = 0;
  if (S.shake > 0) { S.shake -= dt; sx = (Math.random() - 0.5) * S.shake; sy = (Math.random() - 0.5) * S.shake; }
  scene.fog.near = d + 6; scene.fog.far = d + 42;
  // 縦画面は見下ろし気味にして、コートの奥行きを縦方向に使う
  const pitch = camera.aspect >= 1 ? 0.3 : 0.3 + (1 - camera.aspect) * 0.75;
  camera.position.set(S.cam.x + sx, 2.4 + d * pitch + sy, d);
  camera.lookAt(S.cam.x, camera.aspect >= 1 ? 1.3 : 0.6, 0);
}

function drawUI() {
  uictx.clearRect(0, 0, innerWidth, innerHeight);
  if (!S.drag || S.screen !== 'play') return;
  const d = S.drag, aim = aimFromDrag(d);
  const g = uictx;
  g.lineWidth = 2;
  g.strokeStyle = 'rgba(238,243,238,0.35)';
  g.beginPath(); g.arc(d.x0, d.y0, aim.R, 0, Math.PI * 2); g.stroke();
  g.fillStyle = 'rgba(238,243,238,0.7)';
  g.beginPath(); g.arc(d.x0, d.y0, 5, 0, Math.PI * 2); g.fill();
  if (aim.len < 12) return;
  const ang = Math.atan2(d.y - d.y0, d.x - d.x0);
  g.strokeStyle = '#f4b63f'; g.lineWidth = 4; g.lineCap = 'round';
  g.beginPath(); g.moveTo(d.x0, d.y0); g.lineTo(d.x, d.y); g.stroke();
  const hl = 14;
  g.beginPath();
  g.moveTo(d.x, d.y);
  g.lineTo(d.x - hl * Math.cos(ang - 0.45), d.y - hl * Math.sin(ang - 0.45));
  g.moveTo(d.x, d.y);
  g.lineTo(d.x - hl * Math.cos(ang + 0.45), d.y - hl * Math.sin(ang + 0.45));
  g.stroke();
  // パワーの弧
  g.strokeStyle = 'rgba(244,182,63,0.45)'; g.lineWidth = 8;
  g.beginPath(); g.arc(d.x0, d.y0, aim.R + 10, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * Math.min(1, aim.len / aim.R)); g.stroke();
}

function updateHint() {
  if (hintFlash > 0 || S.screen !== 'play' || !S.match) return;
  const m = S.match, h = $('hint');
  if (S.drag) { h.textContent = ''; return; }
  let t = '';
  if (m.phase === 'serve') t = m.server === 'P' ? 'ドラッグしてサーブ。光っている対角のコートへ' : '相手のサーブ';
  else if (m.phase === 'rally') {
    if (m.lastHitter === 'O') t = m.commit ? '予約済み。打点に入ったら自動で打つ' : 'ドラッグで配球を決める（離すと予約）';
    else t = '';
  }
  if (fine && t) t += '　Q/E 面 · W/S コース · Space ジャンプ';
  h.textContent = t;
}

// =====================================================================
// 結果
// =====================================================================
function showResult() {
  const m = S.match;
  S.screen = 'result';
  $('touch').hidden = true;
  const mine = m.stats.hits.filter(h => h.by === 'P');
  const won = m.score.P > m.score.O;
  $('resTitle').textContent = won ? 'WIN' : 'LOSE';
  $('resTitle').style.color = won ? 'var(--you)' : 'var(--cpu)';
  $('resScore').textContent = `${m.score.P} – ${m.score.O} · ${LEVELS[S.levelKey].jp}`;
  const smashes = mine.filter(h => h.name.includes('スマッシュ'));
  $('rsSmash').textContent = smashes.length ? Math.round(Math.max(...smashes.map(h => h.kmh))) + 'km/h' : '—';
  const aces = m.stats.points.filter(p => p.winner === 'P' && p.reason === 'in').length;
  const errs = m.stats.points.filter(p => p.by === 'P' && p.reason !== 'in').length;
  $('rsAce').textContent = aces; $('rsErr').textContent = errs;
  const avg = a => a.reduce((x, y) => x + y, 0) / a.length;
  $('rsTime').textContent = S.decisions.length ? avg(S.decisions).toFixed(2) + 's' : '—';
  const margins = mine.filter(h => !h.serve && h.landIn && isFinite(h.oppMargin)).map(h => Math.max(0, h.oppMargin));
  $('rsMargin').textContent = margins.length ? avg(margins).toFixed(2) + 's' : '—';
  const late = mine.filter(h => h.late).length;
  $('rsLate').textContent = late;

  // 傾向からアドバイスを2つまで
  const tips = [];
  const netSmash = m.stats.points.filter(p => p.by === 'P' && p.reason === 'net' && (p.shot || '').includes('スマッシュ')).length;
  const read = mine.filter(h => h.commitLead > 0.6).length;
  const sliced = mine.filter(h => Math.abs(h.slice) > 0.2).length;
  if (netSmash >= 2) tips.push(['スマッシュがネットに', `${netSmash}本。奥からは −8° 前後が限界、角度を付けたいならジャンプして打点を上げる。`]);
  if (late >= 3) tips.push(['判断遅れ', `${late}回。完璧でなくていいので、シャトルが頂点を越える前に決める。`]);
  if (read >= 4) tips.push(['読まれている', `早すぎる予約が${read}回。相手の反応が速くなる。届く直前まで溜めると効く。`]);
  if (margins.length && avg(margins) > 0.4) tips.push(['相手に余裕', `平均 ${avg(margins).toFixed(2)}秒。前後に揺さぶり、戻る方向の逆へ。`]);
  if (mine.length > 10 && sliced === 0) tips.push(['面を使っていない', 'カットは球が遅く短く曲がり、相手の一歩目を遅らせる。']);
  if (!tips.length) tips.push(['いい試合', '次は難易度を上げるか、マニュアル移動で。']);
  const adv = $('rsAdvice'); adv.innerHTML = '';
  tips.slice(0, 2).forEach(([b, t], i) => {
    if (i) adv.append(document.createTextNode(' '));
    const e = document.createElement('b'); e.textContent = b + '：'; adv.append(e, document.createTextNode(t));
  });
  $('resultScreen').hidden = false;
}

$('startBtn').addEventListener('click', startMatch);
$('againBtn').addEventListener('click', startMatch);
$('menuBtn').addEventListener('click', toTitle);

// 前回の設定
const saved = store.get('arena.settings');
if (saved) {
  const set = (id) => { const el = $(id); if (el) el.checked = true; };
  set('lv-' + saved.lv); set('mv-' + saved.mv); set('len-' + saved.len);
}
if (!fine) { const m = $('mv-auto'); if (m && !saved) m.checked = true; }

setFace(0);
resize();
requestAnimationFrame(frame);

// 動作確認用（URL に ?debug を付けたときだけ）
if (location.search.includes('debug')) window.__arena = S;
