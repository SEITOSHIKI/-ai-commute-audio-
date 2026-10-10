/*
 * 試合進行（描画なし）。ブラウザとテストの両方から同じロジックを動かす。
 *
 * 人間の操作は「配球の予約 (commit)」。シャトルが届く前に、角度・コース・強さ・面（スライス）を決めておくと、
 * 打点に入った瞬間に一番いいタイミングで打つ。早く決めすぎると CPU に読まれ、
 * 決められないまま届くと「判断遅れ」の甘いロブになる。
 */
import {
  C, LEVELS, makePlayer, baseFor, makeShuttle, stepShuttle, simulate, landsIn, launch, nameShot,
  reachQuality, moveToward, steer, stepJump, planIntercept, receiverMargin, chooseShot, serveSetup, hyp2, retarget,
} from './physics.js';

const SUB = C.DT; // 物理の固定ステップ

export class Match {
  constructor(opts = {}) {
    this.level = LEVELS[opts.level || 'club'];
    this.levelKey = opts.level || 'club';
    this.target = opts.target || 11;
    this.cap = this.target === 11 ? 15 : 30;
    this.autoMove = opts.autoMove !== false;
    this.ctrl = { P: opts.human === false ? 'ai' : 'human', O: 'ai' };
    // テスト用: P を AI にしたときの強さ
    this.levelP = LEVELS[opts.levelP || opts.level || 'club'];
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

  startRally() {
    this.pl = { P: makePlayer('P'), O: makePlayer('O') };
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
    this.ai = { P: {}, O: {} };  // AI の移動計画
    this.emit({ type: 'serve-ready', server: this.server });
  }

  // 打つ前のシャトル位置（サーブ時は手元）
  servePoint() {
    const sv = this.pl[this.server];
    return { x: sv.x - sv.side * 0.35, y: 1.0, z: sv.z + 0.15 };
  }

  // ---------- 人間の入力 ----------
  setMove(ix, iz) { this.moveInput = { x: ix, z: iz }; }
  jump(id = 'P') {
    const p = this.pl[id];
    if (p.y === 0 && p.vy === 0 && this.phase === 'rally') p.vy = C.JUMP_V;
  }
  canCommit() {
    if (this.phase === 'serve') return this.server === 'P' && !this.commit;
    if (this.phase !== 'rally' || this.commit) return false;
    return this.lastHitter === 'O';
  }
  commitShot(aim) {
    if (!this.canCommit()) return false;
    this.commit = { aim, at: this.time };
    if (this.phase === 'serve') { this.doServe('P', aim); return true; }
    const pv = this.preview(aim);
    if (pv.sim.end === 'land') this.commit.land = { x: pv.sim.p.x, z: pv.sim.p.z };
    return true;
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

  levelOf(id) { return id === 'P' ? this.levelP : this.level; }

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
      steer(p, mi.x * k * C.SPEED, mi.z * k * C.SPEED, dt);
    } else {
      const plan = this.ai[id];
      if (incoming && plan.target && this.time >= plan.moveAt) {
        moveToward(p, plan.target.x, plan.target.z, dt, human ? 1 : this.levelOf(id).speedMul);
        if (!human && plan.jumpAt && this.time >= plan.jumpAt && p.y === 0) { p.vy = C.JUMP_V; plan.jumpAt = null; }
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
      const score = q + 0.3 * Math.min(this.shuttle.p.y, C.REACH_TOP + p.y) / C.REACH_TOP;
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
    let aim, late = false, commitLead = null;
    if (this.ctrl[id] === 'human') {
      if (this.commit) {
        aim = this.commit.aim; commitLead = this.time - this.commit.at;
        if (this.commit.land) aim = retarget(contact, -p.side, aim, this.commit.land, { jumping: p.y > 0.05 });
      } else {
        // 判断遅れ: 体勢を崩したまま、とりあえず真ん中へ上げる
        aim = { elev: 38, yaw: 0, power: 0.5, slice: 0 }; late = true; q *= 0.45;
      }
    } else {
      aim = chooseShot(p, this.pl[rcvId], contact, this.rng, this.levelOf(id)).aim;
    }
    // スピンネットが回転している間に触ると乱れる
    if (this.shuttle.tumble > 0) q *= 1 - 0.5 * Math.min(1, this.shuttle.tumble);
    const side = -p.side;
    const noise = this.ctrl[id] === 'ai' ? this.levelOf(id).noise : 0;
    const res = launch(contact, side, aim, q, { rng: this.rng, noise, jumping: p.y > 0.05 });
    this.launchShuttle(id, res, false);
    p.swingT = this.time;
    const sim = this.pred;
    const name = nameShot(contact, side, res, sim, aim);

    // 次の受け手の反応: 早い予約は読まれる、スライスと遅い決断は惑わす
    const rLevel = this.levelOf(rcvId);
    let react = rLevel.react;
    if (commitLead != null) react -= Math.min(0.1, Math.max(0, commitLead - 0.35) * 0.25);
    if (commitLead != null && commitLead < 0.12) react += 0.03;
    if (res.band.name === 'over') react += Math.abs(aim.slice) * 0.06;
    this.planFor(rcvId, react);

    const rm = receiverMargin(this.pl[rcvId], sim, this.level.react, this.level.speedMul);
    const hitInfo = {
      by: id, name, kmh: res.speed * 3.6, elev: res.elev, yaw: res.yaw, slice: aim.slice, power: aim.power,
      quality: q, late, commitLead, contactH: contact.y, jump: p.y > 0.05,
      landIn: sim.end === 'land' && landsIn(sim.p, side, null), end: sim.end,
      oppMargin: rm.raw, oppHigh: rm.high, oppHeight: rm.height,
    };
    this.stats.hits.push(hitInfo);
    this.emit({ type: 'hit', ...hitInfo });
    this.commit = null;
    this.ai[id] = { recoverAt: this.time + 0.12 };
  }

  doServe(id, aim) {
    const res = launch(this.servePoint(), -this.pl[id].side, aim, 1, { serve: true, rng: this.rng, noise: this.ctrl[id] === 'ai' ? this.levelOf(id).noise * 0.5 : 0 });
    this.phase = 'rally';
    this.launchShuttle(id, res, true);
    this.pl[id].swingT = this.time;
    const rcvId = this.other(id);
    this.planFor(rcvId, this.levelOf(rcvId).react);
    const sim = this.pred;
    const name = nameShot(this.servePoint(), -this.pl[id].side, res, sim, aim);
    const rm = receiverMargin(this.pl[rcvId], sim, this.level.react, this.level.speedMul);
    const info = { by: id, name, kmh: res.speed * 3.6, elev: res.elev, yaw: res.yaw, slice: 0, power: aim.power, quality: 1, late: false, serve: true, contactH: 1, oppMargin: rm.raw, oppHigh: rm.high, oppHeight: rm.height, end: sim.end };
    this.stats.hits.push(info);
    this.emit({ type: 'hit', ...info });
    this.commit = null;
    this.ai[id] = { recoverAt: this.time + 0.1 };
  }

  launchShuttle(id, res, serve) {
    this.shuttle = res.shuttle;
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
    const r = human ? 0.12 : react;
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

  // 人間が狙いを決める間のプレビュー（予約前の弾道）
  preview(aim) {
    let contact, side = 1, serve = false, jumping = false;
    if (this.phase === 'serve') { contact = this.servePoint(); serve = true; }
    else {
      const p = this.pl.P;
      const it = this.ai.P.intercept;
      contact = it ? { x: it.x, y: Math.min(it.y, C.REACH_TOP + p.y), z: it.z } : { x: p.x + 0.5, y: 2.2, z: p.z };
      jumping = p.y > 0.05 || p.vy > 0;
    }
    const res = launch(contact, side, aim, 1, { serve, jumping });
    return { contact, res, sim: simulate(res.shuttle, { every: 2 }) };
  }
}
