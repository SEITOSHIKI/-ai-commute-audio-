// AI 同士で大量にラリーを回し、ゲームバランスを確認する。
//   node games/badminton/sim-test.js
const E = require('./engine.js');

function playRally(serverId, serverScore, temps, rng, stats) {
  const pl = { P: E.makePlayer('P'), O: E.makePlayer('O') };
  const s = E.serveSetup(serverId, serverScore);
  const rcvId = serverId === 'P' ? 'O' : 'P';
  pl[serverId].lastHit = s.serverPos; pl[serverId].lastHitT = 0;
  pl[rcvId].lastHit = s.receiverPos; pl[rcvId].lastHitT = 0;

  let hitter = serverId, from = s.serverPos, cat = 'SERVE', t = 0, n = 0;
  for (;;) {
    const rcv = hitter === 'P' ? 'O' : 'P';
    const ranked = E.rankOptions(pl[hitter], pl[rcv], from, cat, t, s.serveCol);
    const zone = E.aiChoose(ranked, temps[hitter], rng);
    const shot = E.execute(pl[hitter], pl[rcv], from, cat, zone, t, rng);
    E.commitHit(pl[hitter], from, t);
    n++;
    stats.shots[shot.type] = (stats.shots[shot.type] || 0) + 1;
    stats.cats[cat] = (stats.cats[cat] || 0) + 1;
    if (shot.error) { stats.end['err:' + shot.type] = (stats.end['err:' + shot.type] || 0) + 1; return { winner: rcv, n }; }
    if (shot.winner) { stats.end['win:' + shot.type] = (stats.end['win:' + shot.type] || 0) + 1; return { winner: hitter, n }; }
    t = shot.t0 + shot.F; from = shot.to; cat = shot.rcvCat; hitter = rcv;
    if (n > 200) return { winner: hitter, n };
  }
}

function run(tP, tO, games, seed) {
  const rng = E.mulberry32(seed);
  const stats = { shots: {}, cats: {}, end: {} };
  let rallies = 0, totalLen = 0, pWins = 0, pPts = 0, oPts = 0;
  for (let g = 0; g < games; g++) {
    const sc = { P: 0, O: 0 }; let server = 'P';
    while (!((sc.P >= 11 || sc.O >= 11) && Math.abs(sc.P - sc.O) >= 2) && sc.P < 15 && sc.O < 15) {
      const r = playRally(server, sc[server], { P: tP, O: tO }, rng, stats);
      sc[r.winner]++; server = r.winner; rallies++; totalLen += r.n;
    }
    if (sc.P > sc.O) pWins++;
    pPts += sc.P; oPts += sc.O;
  }
  return { avgLen: totalLen / rallies, pWinRate: pWins / games, pts: [pPts, oPts], stats };
}

const top = o => Object.entries(o).sort((a, b) => b[1] - a[1]).slice(0, 12);
const base = run(0.3, 0.3, 300, 7);
console.log('同レベル: 平均ラリー長', base.avgLen.toFixed(2), '勝率', base.pWinRate.toFixed(2));
console.log('ショット', top(base.stats.shots));
console.log('打点', top(base.stats.cats));
console.log('決着', top(base.stats.end));
for (const [tp, to] of [[0.05, 0.3], [0.05, 0.8], [0.3, 1.5], [3, 0.3]]) {
  const r = run(tp, to, 300, 11);
  console.log(`温度 P=${tp} vs O=${to}: P勝率 ${r.pWinRate.toFixed(2)} 得点 ${r.pts} 平均ラリー ${r.avgLen.toFixed(2)}`);
}

// 診断: ショット種別ごとの margin 分布
if (process.argv[2] === 'diag') {
  const rng = E.mulberry32(3); const m = {};
  for (let i = 0; i < 2000; i++) {
    const pl = { P: E.makePlayer('P'), O: E.makePlayer('O') };
    const s = E.serveSetup('P', 0);
    pl.P.lastHit = s.serverPos; pl.P.lastHitT = 0; pl.O.lastHit = s.receiverPos; pl.O.lastHitT = 0;
    let h = 'P', from = s.serverPos, cat = 'SERVE', t = 0;
    for (let n = 0; n < 60; n++) {
      const r = h === 'P' ? 'O' : 'P';
      const ranked = E.rankOptions(pl[h], pl[r], from, cat, t, s.serveCol);
      const shot = E.execute(pl[h], pl[r], from, cat, E.aiChoose(ranked, 0.3, rng), t, rng);
      E.commitHit(pl[h], from, t);
      (m[shot.type] = m[shot.type] || []).push(shot.margin);
      if (shot.error || shot.winner) break;
      t += shot.F; from = shot.to; cat = shot.rcvCat; h = r;
    }
  }
  for (const [k, v] of Object.entries(m)) {
    v.sort((a, b) => a - b);
    const q = p => v[Math.floor(p * (v.length - 1))].toFixed(2);
    console.log(k.padEnd(10), String(v.length).padStart(6), 'p10', q(0.1), 'p50', q(0.5), 'p90', q(0.9));
  }
}
