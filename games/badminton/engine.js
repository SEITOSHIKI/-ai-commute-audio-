/*
 * ラリーIQ — 配球シミュレーションエンジン
 *
 * 座標系（メートル）: x = コート横方向（画面右が +）、y = 縦方向（ネットが 0）。
 * プレイヤー(P)は y<0（画面下）、相手(O)は y>0（画面上）。シングルスコートを使用。
 *
 * 配球の良し悪しは「受け手が打点に間に合うまでの余裕 (margin)」で決まる。
 *   margin = 飛行時間 − (反応 + 切り返し + 移動時間)
 * 打った側はショット後にホームポジションへ戻り始めるため、
 * 速い球（スマッシュ等）を打つほど自分の戻りが遅れる、という駆け引きが自然に生まれる。
 */
(function (root) {
  'use strict';

  const C = {
    HALF_L: 6.7,      // ネット〜バックライン
    HALF_W: 2.59,     // シングルス サイドライン（中心から）
    SHORT_LINE: 1.98, // ショートサービスライン
    BASE_D: 3.1,      // ホームポジション（ネットからの距離）
    REACT: 0.20,      // 反応時間（スプリットステップ）
    SPEED: 3.8,       // フットワーク速度 m/s
    REC_SPEED: 3.0,   // 打球後の戻り速度 m/s
    REC_DELAY: 0.15,  // 打球後、戻り始めるまで
    REACH: 0.8,       // ラケット＋踏み込みのリーチ
    TURN_PEN: 0.25,   // 逆を突かれた時の切り返しペナルティ（最大）
  };

  // 狙いゾーン: 行(奥/中/前) × 列(画面左/右)
  const ROWS = { back: 5.9, mid: 3.5, front: 1.3 };
  const COLS = { L: -1.6, R: 1.6 };
  const ZONES = ['BL', 'BR', 'ML', 'MR', 'FL', 'FR'];
  const ROW_OF = { B: 'back', M: 'mid', F: 'front' };

  // cls: 受け手の打点を決める球質
  //   high = 高い球 / fast = 速く沈む球 / flat = 平行球 / short = 前に落ちる球
  const SHOTS = {
    clear:     { jp: 'クリアー',       cls: 'high',  f: d => 0.95 + 0.035 * d, err: 0.02 },
    smash:     { jp: 'スマッシュ',     cls: 'fast',  f: d => 0.24 + 0.030 * d, err: 0.06 },
    drop:      { jp: 'ドロップ',       cls: 'short', f: d => 0.60 + 0.050 * d, err: 0.04 },
    halfsmash: { jp: 'ハーフスマッシュ', cls: 'fast', f: d => 0.42 + 0.035 * d, err: 0.04 },
    cut:       { jp: 'カット',         cls: 'short', f: d => 0.55 + 0.050 * d, err: 0.05 },
    longdrive: { jp: 'ロングドライブ', cls: 'flat',  f: d => 0.40 + 0.055 * d, err: 0.04 },
    drive:     { jp: 'ドライブ',       cls: 'flat',  f: d => 0.30 + 0.055 * d, err: 0.03 },
    block:     { jp: 'ブロック',       cls: 'short', f: d => 0.55 + 0.050 * d, err: 0.03 },
    lob:       { jp: 'ロブ',           cls: 'high',  f: d => 0.90 + 0.040 * d, err: 0.02 },
    half:      { jp: 'ハーフ',         cls: 'flat',  f: d => 0.50 + 0.065 * d, err: 0.04 },
    hairpin:   { jp: 'ヘアピン',       cls: 'short', f: d => 0.72 + 0.100 * d, err: 0.05 },
    push:      { jp: 'プッシュ',       cls: 'fast',  f: d => 0.18 + 0.030 * d, err: 0.05 },
    sshort:    { jp: 'ショートサーブ', cls: 'short', f: d => 0.80 + 0.050 * d, err: 0.02, serve: true },
    slong:     { jp: 'ロングサーブ',   cls: 'high',  f: d => 1.10 + 0.045 * d, err: 0.02, serve: true },
    sdrive:    { jp: 'ドライブサーブ', cls: 'flat',  f: d => 0.35 + 0.050 * d, err: 0.06, serve: true },
  };

  // 打点カテゴリ → ゾーン行ごとに打てるショット
  const MENU = {
    HIGH:        { back: 'clear',     mid: 'smash',     front: 'drop',    q: 1.00, jp: '高い打点' },
    HIGH_LATE:   { back: 'clear',     mid: 'halfsmash', front: 'cut',     q: 0.88, jp: '打点遅れ' },
    MID:         { back: 'longdrive', mid: 'drive',     front: 'block',   q: 1.00, jp: '胸の高さ' },
    MID_NET:     { back: 'lob',       mid: 'push',      front: 'hairpin', q: 1.00, jp: 'ネット前・白帯より上' },
    LOW:         { back: 'lob',       mid: 'half',      front: 'hairpin', q: 1.00, jp: '低い打点' },
    LOW_STRETCH: { back: 'lob',       mid: 'half',      front: 'hairpin', q: 0.72, jp: '体勢崩れ' },
    SERVE:       { back: 'slong',     mid: 'sdrive',    front: 'sshort',  q: 1.00, jp: 'サーブ' },
  };

  // 相手に渡した打点カテゴリの価値（打った側から見て）
  const CAT_VALUE = {
    HIGH: -0.7, HIGH_LATE: -0.15, MID: 0, MID_NET: -0.7,
    LOW: 0.35, LOW_STRETCH: 0.7, WINNER: 2.0,
  };

  const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  function makePlayer(id) {
    const side = id === 'P' ? -1 : 1;
    return {
      id, side,
      base: { x: 0, y: side * C.BASE_D },
      lastHit: { x: 0, y: side * C.BASE_D },
      lastHitT: -99,
    };
  }

  // 打球後、ホームへ戻る途中の位置と移動方向
  function posAt(pl, t) {
    const el = t - pl.lastHitT - C.REC_DELAY;
    const p = pl.lastHit;
    if (el <= 0) return { x: p.x, y: p.y, vx: 0, vy: 0, moving: false };
    const dx = pl.base.x - p.x, dy = pl.base.y - p.y;
    const d = Math.hypot(dx, dy);
    if (d < 1e-6) return { x: p.x, y: p.y, vx: 0, vy: 0, moving: false };
    const tr = Math.min(d, el * C.REC_SPEED);
    return {
      x: p.x + dx / d * tr, y: p.y + dy / d * tr,
      vx: dx / d, vy: dy / d, moving: tr < d,
    };
  }

  // 受け手がターゲットに届くまでの所要時間
  function reachTime(rcv, t, target) {
    const p = posAt(rcv, t);
    const d = dist(p, target);
    let turn = 0;
    if (p.moving && d > 1e-6) {
      const ux = (target.x - p.x) / d, uy = (target.y - p.y) / d;
      const dot = p.vx * ux + p.vy * uy;
      turn = C.TURN_PEN * Math.max(0, -dot);
    }
    return { total: C.REACT + turn + Math.max(0, d - C.REACH) / C.SPEED, turn, start: p };
  }

  function zoneTarget(zone, rcvSide, q) {
    const row = ROW_OF[zone[0]];
    let depth = ROWS[row];
    if (row === 'back') depth -= (1 - q) * 3.5; // 崩れた体勢からの奥への球は浅くなる
    return { x: COLS[zone[1]], y: rcvSide * depth, depth };
  }

  function classify(cls, m, depth, type) {
    if (cls === 'high') {
      if (m >= 0.25) return 'HIGH';
      if (m >= -0.05) return 'HIGH_LATE';
      if (m >= -0.3) return 'LOW_STRETCH';
      return 'WINNER';
    }
    if (cls === 'fast') {
      if (m >= 0.2) return 'MID';
      if (m >= 0) return 'LOW';
      if (m >= -0.15) return 'LOW_STRETCH';
      return 'WINNER';
    }
    if (cls === 'flat') {
      // 時間的余裕のある平行球はチャンスボール
      if (m >= 0.45) return depth < 4.0 ? 'MID_NET' : 'HIGH';
      if (m >= 0.2) return 'MID';
      if (m >= 0) return 'LOW';
      if (m >= -0.2) return 'LOW_STRETCH';
      return 'WINNER';
    }
    // short
    const netThr = type === 'sshort' ? 0.75 : 0.4;
    if (m >= netThr && depth < 2.3) return 'MID_NET';
    if (m >= 0.4) return 'MID';
    if (m >= 0) return 'LOW';
    if (m >= -0.3) return 'LOW_STRETCH';
    return 'WINNER';
  }

  /**
   * ショットを計画する（乱数なし）。評価とAI判断、実際の打球の土台。
   * hitter/rcv: プレイヤー, from: 打点, category: 打点カテゴリ, zone: 狙い, t: 打球時刻
   */
  function plan(hitter, rcv, from, category, zone, t, forcedQ) {
    const menu = MENU[category];
    const row = ROW_OF[zone[0]];
    const type = menu[row];
    const shot = SHOTS[type];
    const q = forcedQ != null ? Math.min(forcedQ, menu.q) : menu.q;
    const to = zoneTarget(zone, rcv.side, q);
    const d = dist(from, to);
    const F = shot.f(d) * (1 + (1 - q) * 1.0);
    const rt = reachTime(rcv, t, to);
    const margin = F - rt.total;
    const rcvCat = classify(shot.cls, margin, to.depth, type);
    const errP = shot.err + (1 - q) * 0.35;
    return { type, cls: shot.cls, q, from, to, F, margin, rcvCat, errP, turn: rt.turn, rcvStart: rt.start, zone, category };
  }

  // 打った側から見たショットの価値（ヒューリスティック）
  function evaluate(hitter, p) {
    let v = CAT_VALUE[p.rcvCat] - 1.2 * clamp(p.margin, -0.5, 1.0);
    // 奥へ追い込んだ高い球は、相手の攻撃力が落ちる
    if (p.rcvCat === 'HIGH' || p.rcvCat === 'HIGH_LATE') v += 0.2 * (p.to.depth - 4);
    v -= p.errP * 5;
    // 自分の戻り: 球が相手に届くまでに戻れない距離
    const remaining = Math.max(0, dist(p.from, hitter.base) - C.REC_SPEED * Math.max(0, p.F - C.REC_DELAY));
    v -= remaining * 0.3;
    return v;
  }

  function validZones(category, serveCol) {
    if (category === 'SERVE') return ZONES.filter(z => z[1] === serveCol);
    return ZONES.slice();
  }

  function rankOptions(hitter, rcv, from, category, t, serveCol) {
    return validZones(category, serveCol).map(z => {
      const p = plan(hitter, rcv, from, category, z, t);
      return { zone: z, plan: p, value: evaluate(hitter, p) };
    }).sort((a, b) => b.value - a.value);
  }

  function aiChoose(ranked, temp, rng) {
    if (temp <= 0) return ranked[0].zone;
    const best = ranked[0].value;
    const w = ranked.map(r => Math.exp((r.value - best) / temp));
    const sum = w.reduce((a, b) => a + b, 0);
    let x = rng() * sum;
    for (let i = 0; i < ranked.length; i++) { x -= w[i]; if (x <= 0) return ranked[i].zone; }
    return ranked[ranked.length - 1].zone;
  }

  function gauss(rng) {
    return (rng() + rng() + rng() - 1.5) / 0.5 * 0.5; // 近似正規 (σ≈0.5)
  }

  /**
   * 実際に打つ（狙いのブレ・ミスを反映）。結果の shot は描画にもそのまま使う。
   * extraErr: 難易度によるAIの追加ミス率
   */
  function execute(hitter, rcv, from, category, zone, t, rng, opts) {
    opts = opts || {};
    const p0 = plan(hitter, rcv, from, category, zone, t, opts.forcedQ);
    const spread = 0.25 + (1 - p0.q) * 0.5;
    const to = {
      x: clamp(p0.to.x + gauss(rng) * spread, -C.HALF_W + 0.15, C.HALF_W - 0.15),
      y: p0.to.y + gauss(rng) * spread * 0.8,
    };
    to.y = rcv.side * clamp(Math.abs(to.y), 0.6, C.HALF_L - 0.2);
    to.depth = Math.abs(to.y);
    const d = dist(from, to);
    const F = SHOTS[p0.type].f(d) * (1 + (1 - p0.q) * 1.0);
    const rt = reachTime(rcv, t, to);
    const margin = F - rt.total;
    const rcvCat = classify(p0.cls, margin, to.depth, p0.type);

    let error = null;
    const errP = p0.errP + (opts.extraErr || 0);
    if (rng() < errP) error = rng() < 0.55 ? 'net' : 'out';

    let end = to;
    if (error === 'net') end = { x: (from.x + to.x) / 2, y: 0, depth: 0 };
    if (error === 'out') {
      const outward = rng() < 0.5 ? 'side' : 'long';
      end = outward === 'side'
        ? { x: Math.sign(to.x || 1) * (C.HALF_W + 0.35), y: to.y, depth: to.depth }
        : { x: to.x, y: rcv.side * (C.HALF_L + 0.4), depth: C.HALF_L + 0.4 };
      if (p0.type === 'sshort' || p0.type === 'hairpin' || p0.type === 'drop') {
        end = { x: to.x, y: rcv.side * (C.SHORT_LINE - 0.3), depth: C.SHORT_LINE - 0.3 }; // 短すぎ
        if (!p0.type.startsWith('s')) error = 'net';
      }
    }

    return {
      hitter: hitter.id, receiver: rcv.id,
      type: p0.type, cls: p0.cls, q: p0.q, zone, category,
      from: { x: from.x, y: from.y }, to, end, F,
      t0: t, margin, rcvCat: error ? null : rcvCat, predCat: rcvCat,
      error, winner: !error && rcvCat === 'WINNER',
      turn: rt.turn, rcvStart: rt.start,
    };
  }

  // 打ったら状態を更新
  function commitHit(hitter, from, t) {
    hitter.lastHit = { x: from.x, y: from.y };
    hitter.lastHitT = t;
  }

  // サーブ位置（ラリーポイント。サーバーの得点が偶数なら右コート）
  function serveSetup(serverId, serverScore) {
    const right = serverScore % 2 === 0;
    // P は上を向いているので右 = +x、O は下を向いているので右 = -x
    const sx = serverId === 'P' ? (right ? 1 : -1) : (right ? -1 : 1);
    const ss = serverId === 'P' ? -1 : 1;
    return {
      serverPos: { x: sx * 0.5, y: ss * 2.3 },
      receiverPos: { x: -sx * 0.8, y: -ss * 2.9 },
      serveCol: sx > 0 ? 'L' : 'R', // 対角
    };
  }

  function zoneLabel(zone, category) {
    const row = ROW_OF[zone[0]];
    return SHOTS[MENU[category][row]].jp;
  }

  function zoneName(zone, viewerId) {
    // viewerId 視点の左右（P視点: 画面左 = 左）
    const rowJp = { B: '奥', M: '中', F: '前' }[zone[0]];
    let lr = zone[1];
    if (viewerId === 'O') lr = lr === 'L' ? 'R' : 'L';
    return (lr === 'L' ? '左' : '右') + rowJp;
  }

  // 乱数（シード付き）
  function mulberry32(a) {
    return function () {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  const Engine = {
    C, ROWS, COLS, ZONES, SHOTS, MENU, CAT_VALUE,
    makePlayer, posAt, reachTime, plan, evaluate, rankOptions, aiChoose,
    execute, commitHit, serveSetup, zoneLabel, zoneName, validZones, mulberry32, dist,
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = Engine;
  else root.RallyEngine = Engine;
})(typeof window !== 'undefined' ? window : globalThis);
