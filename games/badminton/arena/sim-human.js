// 人間役のスクリプトで「予約 → 自動で打つ / 判断遅れ」の流れを確かめる
import { Match } from './match.js';
import { mulberry32 } from './physics.js';
const rng = mulberry32(9);
const result = {};
for (const mode of ['early', 'mid', 'never']) {
  const m = new Match({ level: 'club', rng });
  let cpuHitAt = null, hits = 0, late = 0, steps = 0;
  while (m.phase !== 'over' && steps++ < 300000) {
    m.step(1 / 120);
    for (const e of m.events.splice(0)) {
      if (e.type === 'hit' && e.by === 'O') cpuHitAt = m.time;
      if (e.type === 'hit' && e.by === 'P') { hits++; if (e.late) late++; }
    }
    if (m.phase === 'serve' && m.server === 'P') m.commitShot({ target: { x: 5.9, z: m.serveInfo.zSign * 1.3 }, hclass: 'high', slice: 0 });
    if (m.canCommit() && m.phase === 'rally' && mode !== 'never') {
      const wait = mode === 'early' ? 0.05 : 0.35;
      if (m.time - cpuHitAt > wait) {
        // 相手の逆サイドの奥か前へ。高い打点なら沈める
        const o = m.pl.O, c = m.predictedContact();
        const z = o.z > 0 ? -1.9 : 1.9, deep = o.x < 3.2;
        const hclass = c.y > 2.1 ? 'down' : c.y > 1.3 ? 'flat' : deep ? 'high' : 'high';
        m.commitShot({ target: { x: deep ? 5.6 : 1.4, z }, hclass, slice: 0 });
      }
    }
    if (m.phase === 'dead' && m.phaseT > 0.5) m.nextRally();
  }
  result[mode] = { p: m.score.P, hits, late };
  const pts = m.stats.points;
  console.log(mode.padEnd(6), `スコア ${m.score.P}-${m.score.O} 自分の打球 ${hits} 判断遅れ ${late} 失点理由`, Object.entries(pts.filter(p => p.winner === 'O').reduce((a, p) => (a[p.reason + ':' + p.by] = (a[p.reason + ':' + p.by] || 0) + 1, a), {})));
}

// CI 用の最低限の確認: 予約した配球で打てていること、判断した方が判断しないより強いこと
const ok = result.mid.hits > 10 && result.mid.late < result.mid.hits / 2 && result.mid.p > result.never.p;
if (!ok) { console.error('NG: 予約→打球の流れか、判断の有利さが崩れている', result); process.exitCode = 1; }
else console.log('OK');
