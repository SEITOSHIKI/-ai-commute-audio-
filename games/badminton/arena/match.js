/*
 * 試合進行（描画なし）。ブラウザとテストの両方から同じロジックを動かす。
 *
 * 人間の操作は「配球の予約 (commit)」: 相手コートの着地点・弾道の高さ・ラケット面を決めると、
 * 打点に入った瞬間に、その打点から狙いを再現できる角度と初速を逆算して打つ。
 * 早く決めすぎると CPU に読まれ、決められないまま届くと「判断遅れ」の甘いロブになる。
 */
import {
  C, LEVELS, makePlayer, baseFor, stepShuttle, simulate, landsIn, launch, nameShot, solveShot,
  reachQuality, moveToward, steer, stepJump, planIntercept, receiverMargin, chooseShot, serveSetup,
} from './physics.js';

const SUB = C.DT; // 物理の固定ステップ

export class Match {
  /**
   * opts: { level, levelP, target, autoMove, human, rng, attr: { P, O } }
   * attr は roster.toAttr() の結果（未指定なら標準の選手）
   */
  constructor(opts = {}) {
    this.level = LEVELS[opts.level || 'club'];
    this.levelKey = opts.level || 'club';
    this.target = opts.target || 11;
    this.cap = this.target === 11 ? 15 : 30;
    this.autoMove = opts.autoMove !== false;
    this.ctrl = { P: opts.human === false ? 'ai' : 'human', O: 'ai' };
    this.levelP = LEVELS[opts.levelP || opts.level || 'club'];  // P を AI にしたとき（テスト・デモ）
    this.attr = opts.attr || {};
    this.rng = opts.rng || Math.random;
    this.score = { P: 0, O: 0 };
    this.server = 'P';
    this.events = [];
    this.time = 0;
    this.stats = { hits: [], points: [] };
    this.startRally();
  }

  other(id) { return id === 'P' ? 'O' : 'P'; }
  emit(e) { this.events.push({ ...e, time: this.time }); }
  levelOf(id) { return id === 'P' ? this.levelP : this.level; }
  reactOf(id) { return this.levelOf(id).react * this.pl[id].attr.react; }

  startRally() {
    this.pl = { P: makePlayer('P', this.attr.P), O: makePlayer('O', this.attr.O) };
    const s = serveSetup(this.server, this.score[this.server]);
    this.serveInfo = s;
    const sv = this.pl[this.server], rc = this.pl[this.other(this.server)];
    Object.assign(sv, s.server); Object.assign(rc, s.receiver);
    this.phase = 'serve';
    this.phaseT = 0;
    this.shuttle = null;
    this.lastHitter = null;
    this.lastShotServe = false;
    this.pred = null;
    this.rallyHits = 0;
    this.commit = null;          // 人間の予約
    this.ai = { P: {}, O: {} };  // 移動計画
    this.emit({ type: 'serve-ready', server: this.server });
  }

  // サーブ前のシャトル位置（サーバーの手元）
  servePoint() {
    const sv = this.pl[this.server];
    return { x: sv.x - sv.side * 0.35, y: 1.0, z: sv.z + 0.15 };
  }

  // ---------- 人間の入力 ----------
  setMove(ix, iz) { this.moveInput = { x: ix, z: iz }; }
  jump(id = 'P') {
    const p = this.pl[id];
    if (p.y === 0 && p.vy === 0 && this.phase === 'rally') p.vy = p.attr.jumpV;
  }
  canCommit() {
    if (this.phase === 'serve') return this.server === 'P' && !this.commit;
    if (this.phase !== 'rally' || this.commit) return false;
    return this.lastHitter === 'O';
  }
  /** intent: { target: {x,z}（相手コート上の着地点）, hclass: 'down'|'flat'|'mid'|'high', slice: -1..1 } */
  commitShot(intent) {
    if (!this.canCommit()) return false;
    this.commit = { intent, at: this.time };
    if (this.phase === 'serve') this.doServe('P', null, intent);
    return true;
  }

  // 予約前のプレビュー（予測打点から狙いを解く）
  predictedContact() {
    if (this.phase === 'serve') return this.servePoint();
    const p = this.pl.P, it = this.ai.P.intercept;
    if (it) return { x: it.x, y: Math.min(it.y, p.attr.reachTop + p.y + (p.vy > 0 ? 0.4 : 0)), z: it.z };
    return { x: p.x + 0.5, y: 2.2, z: p.z };
  }
  preview(intent) {
    const contact = this.predictedContact();
    const serve = this.phase === 'serve' ? { zSign: this.serveInfo.zSign } : null;
    const p = this.pl.P;
    const sol = solveShot(contact, 1, intent.target, intent.hclass, intent.slice, { attr: p.attr, serve: !!serve, jumping: p.y > 0.05 || p.vy > 0 });
    const res = launch(contact, 1, sol.aim, 1, { attr: p.attr, serve: !!serve, jumping: p.y > 0.05 });
    const sim = simulate(res.shuttle, { every: 2 });
    return { contact, sol, res, sim, name: nameShot(contact, 1, res, sim, sol.aim) };
  }

  // ---------- 進行 ----------
  step(dt) {
    let left = dt;
    while (left > 1e-9) {
      const h = Math.min(SUB, left);
      this.substep(h);
      left -= h;
    }
  }

  substep(dt) {
    this.time += dt;
    this.phaseT += dt;
    if (this.phase === 'serve') {
      if (this.ctrl[this.server] === 'ai' && this.phaseT > 1.1) {
        const sv = this.pl[this.server];
        const ch = chooseShot(sv, this.pl[this.other(this.server)], this.servePoint(), this.rng,
          this.levelOf(this.server), { serve: { zSign: this.serveInfo.zSign } });
        this.doServe(this.server, ch.aim);
      }
      return;
    }
    if (this.phase === 'dead' || this.phase === 'over') {
      if (this.shuttle) this.fallShuttle(dt);
      for (const id of ['P', 'O']) { const p = this.pl[id]; steer(p, 0, 0, dt); stepJump(p, dt); }
      return;
    }

    // --- rally ---
    const s = this.shuttle;
    const prevX = s.p.x, prevY = s.p.y;
    stepShuttle(s, dt);
    if (s.tumble > 0) s.tumble = Math.max(0, s.tumble - dt);

    // ネット
    if (Math.sign(prevX) !== Math.sign(s.p.x) && prevX !== 0) {
      const f = prevX / (prevX - s.p.x);
      const yAt = prevY + (s.p.y - prevY) * f;
      if (yAt < C.NET_H) {
        s.p.x = -Math.sign(s.p.x) * 0.05; s.v.x *= -0.1; s.v.z *= 0.2; s.v.y = Math.min(s.v.y, 0);
        return this.endRally(this.other(this.lastHitter), 'net');
      }
    }
    // 着地
    if (s.p.y <= 0) {
      s.p.y = 0;
      const hitterSide = this.pl[this.lastHitter].side * -1;
      const inside = landsIn(s.p, hitterSide, this.lastShotServe ? { zSign: this.serveInfo.zSign } : null);
      return this.endRally(inside ? this.lastHitter : this.other(this.lastHitter), inside ? 'in' : 'out');
    }

    for (const id of ['P', 'O']) this.updatePlayer(id, dt);
  }

  updatePlayer(id, dt) {
    const p = this.pl[id];
    stepJump(p, dt);
    const incoming = this.lastHitter && this.lastHitter !== id;
    const human = this.ctrl[id] === 'human';

    // 移動
    if (human && !this.autoMove) {
      const mi = this.moveInput || { x: 0, z: 0 };
      const m = Math.hypot(mi.x, mi.z);
      const k = m > 1 ? 1 / m : 1;
      steer(p, mi.x * k * p.attr.speed, mi.z * k * p.attr.speed, dt);
    } else {
      const plan = this.ai[id];
      if (incoming && plan.target && this.time >= plan.moveAt) {
        moveToward(p, plan.target.x, plan.target.z, dt, human ? 1 : this.levelOf(id).speedMul);
        if (!human && plan.jumpAt && this.time >= plan.jumpAt && p.y === 0) { p.vy = p.attr.jumpV; plan.jumpAt = null; }
      } else if (!incoming || !plan.target) {
        const b = baseFor(p);
        if (this.time >= (plan.recoverAt || 0)) moveToward(p, b.x, b.z, dt, 0.8);
        else steer(p, 0, 0, dt);
      } else steer(p, 0, 0, dt);
    }

    // 打つ
    if (!incoming) return;
    const q = reachQuality(p, this.shuttle.p);
    const st = this.ai[id];
    if (q > 0) {
      // 「打点の質 + 高さ」が最大になった瞬間に打つ（高い打点ほど攻められる）
      const score = q + 0.3 * Math.min(this.shuttle.p.y, p.attr.reachTop + p.y) / p.attr.reachTop;
      const past = st.prevScore != null && score < st.prevScore - 1e-4;
      if (past || this.shuttle.p.y < 0.32) this.hit(id, Math.max(q, st.prevQ || 0));
      else { st.prevQ = q; st.prevScore = score; }
    } else if (st.prevQ) {
      this.hit(id, st.prevQ); // 圏外に出る瞬間
    }
  }

  hit(id, q) {
    const p = this.pl[id];
    const rcvId = this.other(id);
    const contact = { ...this.shuttle.p };
    const side = -p.side;
    const jumping = p.y > 0.05;
    const incomingSpeed = Math.hypot(this.shuttle.v.x, this.shuttle.v.y, this.shuttle.v.z);
    // 守備力: 速い球ほど受ける上手さが効く
    if (incomingSpeed > 20) q = Math.min(1, q + 0.3 * p.attr.defense - 0.14);
    // スピンネットが回転している間に触ると乱れる
    if (this.shuttle.tumble > 0) q *= 1 - 0.5 * Math.min(1, this.shuttle.tumble);

    let aim, late = false, commitLead = null, downgraded = null, intent = null;
    if (this.ctrl[id] === 'human') {
      if (this.commit) {
        intent = this.commit.intent;
        commitLead = this.time - this.commit.at;
        const sol = solveShot(contact, side, intent.target, intent.hclass, intent.slice, { attr: p.attr, jumping });
        aim = sol.aim; downgraded = sol.downgraded;
      } else {
        // 判断遅れ: 体勢を崩したまま、とりあえず真ん中へ上げる
        aim = { elev: 38, yaw: 0, power: 0.5, slice: 0 }; late = true; q *= 0.45;
      }
    } else {
      aim = chooseShot(p, this.pl[rcvId], contact, this.rng, this.levelOf(id)).aim;
    }
    const noise = this.ctrl[id] === 'ai' ? this.levelOf(id).noise : 0.08;
    const res = launch(contact, side, aim, q, { rng: this.rng, noise, jumping, attr: p.attr });
    this.launchShuttle(id, res, false);
    const sim = this.pred;
    const name = nameShot(contact, side, res, sim, aim);
    const swing = swingFor(p, contact, res, name, late);
    p.swingT = this.time; p.swing = swing;

    // 次の受け手の反応: 早い予約は読まれる、スライスと遅い決断は惑わす
    let react = this.reactOf(rcvId);
    if (commitLead != null) react -= Math.min(0.1, Math.max(0, commitLead - 0.35) * 0.25);
    if (commitLead != null && commitLead < 0.12) react += 0.03;
    if (res.band.name === 'over') react += Math.abs(aim.slice) * p.attr.deceive;
    this.planFor(rcvId, react);

    const rm = receiverMargin(this.pl[rcvId], sim, this.level.react, this.level.speedMul);
    const hitInfo = {
      by: id, name, kmh: res.speed * 3.6, elev: res.elev, yaw: res.yaw, slice: aim.slice || 0,
      quality: q, late, commitLead, contactH: contact.y, jump: jumping, swing,
      intent, downgraded, landIn: sim.end === 'land' && landsIn(sim.p, side, null), end: sim.end,
      land: sim.p, oppMargin: rm.raw, oppHigh: rm.high, oppHeight: rm.height,
    };
    this.stats.hits.push(hitInfo);
    this.emit({ type: 'hit', ...hitInfo });
    this.commit = null;
    this.ai[id] = { recoverAt: this.time + 0.12 };
  }

  doServe(id, aim, intent) {
    const p = this.pl[id];
    const contact = this.servePoint();
    if (intent) aim = solveShot(contact, -p.side, intent.target, intent.hclass, intent.slice, { attr: p.attr, serve: true }).aim;
    const res = launch(contact, -p.side, aim, 1, { serve: true, rng: this.rng, noise: this.ctrl[id] === 'ai' ? this.levelOf(id).noise * 0.5 : 0.05, attr: p.attr });
    this.phase = 'rally';
    this.launchShuttle(id, res, true);
    const rcvId = this.other(id);
    this.planFor(rcvId, this.reactOf(rcvId));
    const sim = this.pred;
    const name = nameShot(contact, -p.side, res, sim, aim);
    const swing = { type: name === 'ロングサーブ' ? 'serve_long' : 'serve_short', hand: name === 'ロングサーブ' ? 'fh' : 'bh' };
    p.swingT = this.time; p.swing = swing;
    const rm = receiverMargin(this.pl[rcvId], sim, this.level.react, this.level.speedMul);
    const info = { by: id, name, kmh: res.speed * 3.6, elev: res.elev, yaw: res.yaw, slice: 0, quality: 1, late: false, serve: true, contactH: 1, swing, intent, land: sim.p, oppMargin: rm.raw, oppHigh: rm.high, oppHeight: rm.height, end: sim.end };
    this.stats.hits.push(info);
    this.emit({ type: 'hit', ...info });
    this.commit = null;
    this.ai[id] = { recoverAt: this.time + 0.1 };
  }

  launchShuttle(id, res, serve) {
    this.shuttle = res.shuttle;
    this.launchT = this.time;
    this.lastHitter = id;
    this.lastShotServe = serve;
    this.rallyHits++;
    this.pred = simulate(this.shuttle, { every: 3 });
    this.ai[this.other(id)] = {};
  }

  // 受け手の迎撃計画（人間のオート移動もこれを使う）
  planFor(id, react) {
    const p = this.pl[id];
    const human = this.ctrl[id] === 'human';
    const r = human ? 0.12 * p.attr.react : react;
    const it = planIntercept(p, this.pred, 0, r, human ? 1 : this.levelOf(id).speedMul, true);
    const plan = { moveAt: this.time + r, prevQ: null };
    if (it) {
      plan.target = { x: it.x + p.side * 0.45, z: it.z };
      plan.intercept = it;
      if (it.jump && !human) plan.jumpAt = this.time + Math.max(0, it.t - 0.3);
    } else {
      // 届かない: 着地点に向かって飛びつく
      plan.target = { x: this.pred.p.x + p.side * 0.3, z: this.pred.p.z };
    }
    this.ai[id] = plan;
  }

  fallShuttle(dt) {
    const s = this.shuttle;
    if (s.p.y > 0) { stepShuttle(s, dt); if (s.p.y < 0) s.p.y = 0; }
  }

  endRally(winner, reason) {
    this.phase = 'dead';
    this.phaseT = 0;
    this.score[winner]++;
    const last = this.stats.hits[this.stats.hits.length - 1];
    const pt = { winner, reason, by: this.lastHitter, shot: last && last.name, rally: this.rallyHits, kmh: last && last.kmh };
    this.stats.points.push(pt);
    this.emit({ type: 'point', ...pt, score: { ...this.score } });
    this.server = winner;
    if (this.isOver()) { this.phase = 'over'; this.emit({ type: 'game', winner, score: { ...this.score } }); }
  }

  isOver() {
    const a = this.score.P, b = this.score.O;
    if (a === this.cap || b === this.cap) return true;
    return (a >= this.target || b >= this.target) && Math.abs(a - b) >= 2;
  }

  nextRally() { if (this.phase === 'dead') this.startRally(); }
}

/**
 * ショットに合ったスイングの型。描画側はこれでフォームを切り替える。
 * hand: 'fh' フォア / 'bh' バック（利き手と打点の左右で決まる）
 */
export function swingFor(p, contact, res, name, late) {
  // 自分の進行方向に対する打点の左右。右利きで +x を向く P は +z がフォア側
  const facing = -p.side; // +1: +x を向いている
  const racketSide = (p.attr.lefty ? -1 : 1) * facing;
  const lateral = (contact.z - p.z) * racketSide;
  const hand = lateral > -0.12 ? 'fh' : 'bh';
  const band = res.band.name;
  let type;
  if (late) type = 'lift';
  else if (band === 'over') {
    if (name.includes('スマッシュ')) type = p.y > 0.05 ? 'jumpsmash' : 'smash';
    else if (name === 'ドロップ' || name === 'カット') type = 'drop';
    else type = 'clear';
  } else if (band === 'side') type = name === 'プッシュ' ? 'push' : 'drive';
  else if (name === 'ヘアピン' || name === 'スピンネット') type = 'net';
  else type = 'lift';
  const lunge = Math.abs(contact.x) < 2.6 && contact.y < 1.3;
  return { type, hand, lunge, height: contact.y };
}
