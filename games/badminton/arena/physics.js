/*
 * ラリーIQ アリーナ — 物理と CPU の頭脳（描画なし。Node でもテストできる）
 *
 * 座標（メートル）: x = コートの長さ方向（ネットが 0、YOU は x<0 側、CPU は x>0 側）
 *                   y = 高さ、z = コートの幅方向（シングルス ±2.59）
 * 時間は「ゲーム内の実時間（秒）」。画面の速さは難易度ごとのタイムスケールで変える。
 *
 * シャトルの空気抵抗は a = g − (g / Vt²)|v|v（終端速度 Vt ≈ 6.8 m/s）。
 * 初速 300km/h 超のスマッシュでも急減速し、クリアーは奥でストンと落ちる、実物の軌道になる。
 */

export const C = {
  G: 9.81,
  VT: 6.8,            // シャトルの終端速度
  L: 6.7,             // ネット〜バックライン
  W: 2.59,            // シングルス サイドライン
  SHORT: 1.98,        // ショートサービスライン
  NET_H: 1.524,       // ネット中央の高さ
  DT: 1 / 240,
  SPEED: 4.6,         // フットワーク最高速度 m/s
  ACC: 26,            // 加速度
  REACH: 1.05,        // 体の中心からラケットまでの水平リーチ
  REACH_TOP: 2.55,    // 立ったままのラケット最高到達点
  REACH_LOW: 0.12,
  JUMP_V: 3.1,        // ジャンプ初速（約0.5m跳ぶ）
  BASE_X: 3.0,        // ホームポジション（ネットからの距離）
  SERVE_MAX_H: 1.15,  // サーブの打点上限（ルール）
  SLICE_K: 0.012,     // スライスによる横方向の曲がり
  SLICE_TAU: 0.45,    // 曲がりが減衰する時定数
  SLICE_SPEED_LOSS: 0.32,
};

const K_DRAG = C.G / (C.VT * C.VT);
const deg = d => d * Math.PI / 180;
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const lerp = (a, b, t) => a + (b - a) * t;
export const hyp2 = (x, z) => Math.hypot(x, z);

// ---------------- シャトル ----------------

export function makeShuttle(p, v, slice = 0, tumble = 0) {
  return { p: { ...p }, v: { ...v }, slice, tumble, t: 0 };
}

export function stepShuttle(s, dt) {
  const { v } = s;
  const sp = Math.hypot(v.x, v.y, v.z);
  let ax = -K_DRAG * sp * v.x;
  let ay = -C.G - K_DRAG * sp * v.y;
  let az = -K_DRAG * sp * v.z;
  if (s.slice) {
    // 面を切った分だけ横に曲がる（初速が高いほど強く、すぐ減衰）
    const h = Math.hypot(v.x, v.z) || 1;
    const lat = s.slice * C.SLICE_K * sp * sp * Math.exp(-s.t / C.SLICE_TAU);
    ax += (-v.z / h) * lat;
    az += (v.x / h) * lat;
  }
  v.x += ax * dt; v.y += ay * dt; v.z += az * dt;
  s.p.x += v.x * dt; s.p.y += v.y * dt; s.p.z += v.z * dt;
  s.t += dt;
}

/**
 * 着地 or ネットまで飛ばす。path は描画やAIの到達判定に使うサンプル列。
 * 戻り値: { end: 'land'|'net', t, p, path: [{t,x,y,z}], clearance }
 */
export function simulate(shuttle, opts = {}) {
  const s = makeShuttle(shuttle.p, shuttle.v, shuttle.slice, shuttle.tumble);
  s.t = shuttle.t || 0;
  const t0 = s.t;
  const dt = opts.dt || C.DT;
  const every = opts.every || 4;
  const maxT = opts.maxT || 6;
  const path = [{ t: 0, x: s.p.x, y: s.p.y, z: s.p.z }];
  let clearance = null;
  for (let i = 1; s.t - t0 < maxT; i++) {
    const px = s.p.x, py = s.p.y;
    stepShuttle(s, dt);
    if (Math.sign(px) !== Math.sign(s.p.x) && px !== 0) {
      const f = px / (px - s.p.x);
      const yAt = py + (s.p.y - py) * f;
      clearance = yAt - C.NET_H;
      if (yAt < C.NET_H) {
        return { end: 'net', t: s.t - t0, p: { x: 0, y: yAt, z: s.p.z }, path, clearance };
      }
    }
    if (s.p.y <= 0) {
      const p = { x: s.p.x, y: 0, z: s.p.z };
      path.push({ t: s.t - t0, ...p });
      return { end: 'land', t: s.t - t0, p, path, clearance };
    }
    if (i % every === 0) path.push({ t: s.t - t0, x: s.p.x, y: s.p.y, z: s.p.z });
  }
  return { end: 'land', t: s.t - t0, p: { ...s.p }, path, clearance };
}

// 着地点が相手コート内か（side = 打った側。+1 は YOU が x+ 方向へ打つ）
export function landsIn(p, side, serve) {
  const x = p.x * side;
  if (x <= 0 || x > C.L || Math.abs(p.z) > C.W) return false;
  if (serve) {
    if (x < C.SHORT) return false;
    if (Math.sign(p.z) !== serve.zSign) return false;
  }
  return true;
}

// ---------------- ショット ----------------

// 打点の高さで打てる角度と最高初速が決まる
export function contactBand(h, jumping) {
  if (h >= 2.0) return { name: 'over', minE: -38, maxE: 62, vmax: jumping ? 98 : 86 };
  if (h >= 1.1) return { name: 'side', minE: -16, maxE: 55, vmax: 56 };
  return { name: 'under', minE: -6, maxE: 72, vmax: 42 };
}

/**
 * 入力パラメータから打球の初速ベクトルを作る。
 * aim: { elev(度), yaw(-1..1 = 幅方向), power(0..1), slice(-1..1) }
 * side: 打つ方向（+1 / -1）, q: 打点の質（0..1）, rng があれば質に応じてブレる
 */
export function launch(contact, side, aim, q, opts = {}) {
  const band = opts.serve ? { name: 'serve', minE: -4, maxE: 62, vmax: 34 } : contactBand(contact.y, opts.jumping);
  const rng = opts.rng;
  const noise = (sd) => rng ? (rng() + rng() + rng() - 1.5) * 2 * sd : 0;
  const errScale = (1 - q) + (opts.noise || 0) + Math.abs(aim.slice) * 0.15;
  const elev = clamp(aim.elev, band.minE, band.maxE) + noise(6 * errScale);
  const yaw = clamp(aim.yaw, -1, 1) * 21 + noise(4 * errScale);
  const vmin = 3.2;
  let speed = lerp(vmin, band.vmax, clamp(aim.power, 0, 1));
  speed *= (1 - C.SLICE_SPEED_LOSS * Math.abs(aim.slice)) * (0.78 + 0.22 * q);
  speed *= 1 + noise(0.05 * errScale);
  const e = deg(elev), y = deg(yaw);
  const v = {
    x: side * Math.cos(e) * Math.cos(y) * speed,
    y: Math.sin(e) * speed,
    z: Math.cos(e) * Math.sin(y) * speed,
  };
  // ネット前の低速スライス（スピンネット）は回転して返しにくい
  const nearNet = Math.abs(contact.x) < 2.4 && speed < 16;
  const tumble = nearNet && Math.abs(aim.slice) > 0.2 ? 0.7 * Math.abs(aim.slice) : 0;
  const s = makeShuttle(contact, v, aim.slice * side, tumble);
  return { shuttle: s, band, elev, yaw, speed };
}

/**
 * 予約した「狙い（着地点と強さ）」を、実際の打点から再現できる角度に合わせ直す。
 * 打点が想定より低ければ角度は上向きに補正される。補正しきれなければネット／アウトのまま。
 */
export function retarget(contact, side, aim, land, opts = {}) {
  let best = aim, bestD = Infinity;
  for (let de = -16; de <= 16; de += 1) {
    const a = { ...aim, elev: aim.elev + de };
    const r = launch(contact, side, a, 1, opts);
    const s = simulate(r.shuttle, { dt: 1 / 120, every: 8 });
    const d = s.end === 'land' ? hyp2(s.p.x - land.x, s.p.z - land.z) + Math.abs(de) * 0.01 : 50 + Math.abs(de);
    if (d < bestD) { bestD = d; best = a; }
  }
  return best;
}

// 弾道からショット名を付ける（表示・ログ用）
export function nameShot(contact, side, res, sim, aim) {
  const sliced = Math.abs(aim.slice) >= 0.35;
  const landX = sim.p.x * side;
  const near = Math.abs(contact.x) < 2.3;
  const kmh = res.speed * 3.6;
  if (res.band.name === 'serve') return landX > 4.5 ? 'ロングサーブ' : 'ショートサーブ';
  if (res.band.name === 'over') {
    if (res.elev < -4 && kmh > 160) return sliced ? 'カットスマッシュ' : 'スマッシュ';
    if (landX < 3.8) return sliced ? 'カット' : 'ドロップ';
    if (res.elev < 4 && kmh > 120) return 'ドライブ';
    if (res.elev < 22) return 'ドリブンクリアー';
    return 'クリアー';
  }
  if (near && res.elev < -3 && kmh > 60) return 'プッシュ';
  if (near && landX < 2.6) return sliced ? 'スピンネット' : 'ヘアピン';
  if (res.elev > 22 && landX > 4.2) return 'ロブ';
  if (Math.abs(res.elev) < 14 && kmh > 70) return 'ドライブ';
  if (landX < 3.6) return near ? 'ヘアピン' : 'ハーフ';
  return 'ロブ';
}

// ---------------- 選手 ----------------

export function makePlayer(id) {
  const side = id === 'P' ? -1 : 1; // 自陣の符号
  return {
    id, side,
    x: side * C.BASE_X, z: 0, y: 0, vx: 0, vz: 0, vy: 0,
    swingT: -9, armed: null, lastQ: 1,
  };
}

export function baseFor(pl) { return { x: pl.side * C.BASE_X, z: 0 }; }

// 選手がシャトル位置に届くか & 打点の質
export function reachQuality(pl, sp) {
  if (Math.sign(sp.x) !== pl.side && Math.abs(sp.x) > 0.02) return 0;
  const d = hyp2(sp.x - pl.x, sp.z - pl.z);
  const top = C.REACH_TOP + pl.y;
  if (sp.y < C.REACH_LOW || sp.y > top + 0.05) return 0;
  const R = C.REACH + (Math.hypot(pl.vx, pl.vz) > 2.5 ? 0.25 : 0); // 踏み込み
  if (d > R) return 0;
  let q = 1 - 0.55 * (d / R) ** 2;
  // 体の真上・真横すぎる打点や、頭上の限界ギリギリは質が落ちる
  if (sp.y > top - 0.12) q -= 0.15;
  if (sp.y < 0.35) q -= 0.2;
  return clamp(q, 0.25, 1);
}

// 選手を目標地点へ動かす（加速度と最高速度つき）
export function moveToward(pl, tx, tz, dt, speedMul = 1) {
  const dx = tx - pl.x, dz = tz - pl.z;
  const d = Math.hypot(dx, dz);
  const vmax = C.SPEED * speedMul;
  const want = d < 0.05 ? 0 : Math.min(vmax, Math.sqrt(2 * C.ACC * 0.6 * d));
  const wx = d > 1e-6 ? dx / d * want : 0, wz = d > 1e-6 ? dz / d * want : 0;
  steer(pl, wx, wz, dt);
}

export function steer(pl, wx, wz, dt) {
  const ax = wx - pl.vx, az = wz - pl.vz;
  const a = Math.hypot(ax, az), lim = C.ACC * dt;
  const f = a > lim ? lim / a : 1;
  pl.vx += ax * f; pl.vz += az * f;
  pl.x += pl.vx * dt; pl.z += pl.vz * dt;
  // ネットは越えられない
  if (pl.side < 0) pl.x = clamp(pl.x, -C.L - 1.2, -0.35);
  else pl.x = clamp(pl.x, 0.35, C.L + 1.2);
  pl.z = clamp(pl.z, -C.W - 0.9, C.W + 0.9);
}

export function stepJump(pl, dt) {
  if (pl.y > 0 || pl.vy > 0) {
    pl.vy -= C.G * dt; pl.y += pl.vy * dt;
    if (pl.y <= 0) { pl.y = 0; pl.vy = 0; }
  }
}

/**
 * 迎撃プラン: 飛んでくる弾道 path のうち、届く点の中で「一番攻撃的に打てる点」を選ぶ。
 * react: 反応時間, now: path 上の経過時間（すでに飛んだ分）
 */
export function planIntercept(pl, sim, now, react, speedMul = 1, preferHigh = true) {
  let best = null;
  const vmax = C.SPEED * speedMul;
  for (const s of sim.path) {
    if (s.t < now) continue;
    if (Math.sign(s.x) !== pl.side) continue;
    if (s.y < 0.25 || s.y > C.REACH_TOP + 0.45) continue;
    const d = hyp2(s.x - pl.x, s.z - pl.z);
    const need = react + Math.max(0, d - C.REACH * 0.8) / vmax;
    const margin = (s.t - now) - need;
    if (margin < -0.02) continue;
    // 高い打点ほど攻められる。余裕もある程度ほしい
    const h = Math.min(s.y, C.REACH_TOP + 0.4);
    const score = (preferHigh ? h * 0.45 : 0) + Math.min(margin, 0.4) * 1.2;
    if (!best || score > best.score) best = { ...s, margin, score, jump: s.y > C.REACH_TOP - 0.05 };
  }
  return best;
}

/**
 * 受け手から見た「最善の迎撃の余裕」。打った側の評価に使う（小さいほど良い配球）。
 * 届かなければマイナス（=エース級）。アウトの球なら null。
 */
export function receiverMargin(rcv, sim, react, speedMul = 1) {
  let best = -Infinity, bestH = 0, high = -Infinity, raw = -Infinity;
  const vmax = C.SPEED * speedMul;
  for (const s of sim.path) {
    if (Math.sign(s.x) !== rcv.side) continue;
    if (s.y < 0.2 || s.y > C.REACH_TOP + 0.45) continue;
    const d = hyp2(s.x - rcv.x, s.z - rcv.z);
    const m = s.t - (react + Math.max(0, d - C.REACH) / vmax);
    if (m > raw) raw = m;
    if (s.y >= 1.6 && m > high) high = m;
    // 高い打点で取れるほど相手は攻められる → 価値を少し上乗せ
    const v = Math.min(m, 0.6) + (s.y > 2.1 ? 0.18 : s.y > 1.4 ? 0.06 : 0);
    if (v > best) { best = v; bestH = s.y; }
  }
  // margin: 評価値 / raw: 一番余裕のある迎撃 / high: 高い打点（1.6m以上）で取れる余裕
  return { margin: best, height: bestH, raw, high };
}

// ---------------- CPU の配球選択 ----------------

const ELEVS = [-34, -22, -12, -5, 2, 10, 18, 28, 40, 52];
const YAWS = [-1, -0.45, 0, 0.45, 1];
const POWERS = [0.12, 0.3, 0.5, 0.75, 1];
const SLICES = [0, 0.6, -0.6];

/**
 * 候補ショットを物理で全部シミュレートし、相手の余裕が最小になる配球を選ぶ。
 * level: { temp, noise, safety, react }
 */
export function chooseShot(hitter, rcv, contact, rng, level, opts = {}) {
  const side = -hitter.side;
  const cands = [];
  const band = opts.serve ? { minE: -4, maxE: 62 } : contactBand(contact.y, hitter.y > 0.05);
  for (const elev of ELEVS) {
    if (elev < band.minE - 4 || elev > band.maxE + 4) continue;
    for (const yaw of YAWS) for (const power of POWERS) for (const slice of SLICES) {
      if (slice && power < 0.2 && Math.abs(contact.x) > 2.4) continue;
      const aim = { elev, yaw, power, slice };
      const res = launch(contact, side, aim, 1, { serve: opts.serve, jumping: hitter.y > 0.05 });
      const sim = simulate(res.shuttle, { dt: 1 / 120, every: 3 });
      if (sim.end !== 'land' || !landsIn(sim.p, side, opts.serve)) continue;
      // ライン際・ネットすれすれは実行のブレで失点しやすい
      const lx = sim.p.x * side;
      const edge = Math.min(C.W - Math.abs(sim.p.z), C.L - lx, opts.serve ? lx - C.SHORT : 9);
      const clear = sim.clearance == null ? 9 : sim.clearance;
      if (edge < level.safety || clear < level.safety * 0.5) continue;
      const rm = receiverMargin(rcv, sim, level.oppReact ?? 0.22);
      let value = -rm.margin;
      // 打った後に自分がホームへ戻れるか
      const bx = hitter.side * C.BASE_X;
      const own = Math.max(0, hyp2(contact.x - bx, contact.z) / C.SPEED - sim.t * 0.8);
      value -= own * 0.35;
      // 甘い球（相手が高い打点で余裕を持って取れる）を嫌う
      if (rm.height > 2.1 && rm.margin > 0.3) value -= 0.25;
      cands.push({ aim, value, sim, res, rm });
    }
  }
  if (!cands.length) {
    // 安全策: 高いロブ／クリアー
    return { aim: { elev: 45, yaw: 0, power: 0.7, slice: 0 }, value: -9, fallback: true };
  }
  cands.sort((a, b) => b.value - a.value);
  if (!rng || level.temp <= 0) return cands[0];
  const top = cands.slice(0, 40);
  const w = top.map(c => Math.exp((c.value - top[0].value) / level.temp));
  let r = rng() * w.reduce((a, b) => a + b, 0);
  for (let i = 0; i < top.length; i++) { r -= w[i]; if (r <= 0) return top[i]; }
  return top[0];
}

// サーブ位置: サーバーの得点が偶数なら右コート
export function serveSetup(serverId, serverScore) {
  const right = serverScore % 2 === 0;
  // YOU（+x を向く）の右手側は +z、CPU（-x を向く）の右手側は -z
  const zs = serverId === 'P' ? (right ? 1 : -1) : (right ? -1 : 1);
  const xs = serverId === 'P' ? -1 : 1;
  return {
    server: { x: xs * 2.5, z: zs * 0.6 },
    receiver: { x: -xs * 2.9, z: -zs * 0.7 },
    zSign: -zs, // 対角の相手サービスコート
  };
}

export function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

export const LEVELS = {
  beginner: { jp: '入門',   time: 0.42, react: 0.34, speedMul: 0.86, temp: 0.22, noise: 0.35, safety: 0.45 },
  club:     { jp: '部活',   time: 0.52, react: 0.27, speedMul: 0.93, temp: 0.12, noise: 0.22, safety: 0.35 },
  expert:   { jp: '実業団', time: 0.64, react: 0.21, speedMul: 1.0,  temp: 0.06, noise: 0.13, safety: 0.25 },
  pro:      { jp: '代表',   time: 0.8,  react: 0.17, speedMul: 1.05, temp: 0.03, noise: 0.07, safety: 0.16 },
};
