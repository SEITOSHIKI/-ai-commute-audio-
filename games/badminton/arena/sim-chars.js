// 選手ごとの個体差が試合結果に出るかを見る: node games/badminton/arena/sim-chars.js
import { Match } from './match.js';
import { mulberry32 } from './physics.js';
import { CHARACTERS, toAttr } from './roster.js';
const mem = new Map();
const store = { get: k => mem.get(k) ?? null, set: (k, v) => mem.set(k, v) };
const ch = id => CHARACTERS.find(c => c.id === id);
function series(a, b, n) {
  let w = 0, pts = [0, 0], smashA = [], smashB = [];
  for (let i = 0; i < n; i++) {
    const m = new Match({ human: false, level: 'expert', levelP: 'expert', rng: mulberry32(70 + i), attr: { P: toAttr(ch(a)), O: toAttr(ch(b)) } });
    let g = 0;
    while (m.phase !== 'over' && g++ < 400000) { m.step(1 / 60); if (m.phase === 'dead' && m.phaseT > 0.8) m.nextRally(); }
    if (m.score.P > m.score.O) w++;
    pts[0] += m.score.P; pts[1] += m.score.O;
    for (const h of m.stats.hits) if (h.name.includes('スマッシュ')) (h.by === 'P' ? smashA : smashB).push(h.kmh);
  }
  const mx = a => a.length ? Math.round(Math.max(...a)) : '-';
  console.log(`${ch(a).name} vs ${ch(b).name}: ${w}/${n}勝 得点 ${pts.join('-')} 最速スマッシュ ${mx(smashA)} / ${mx(smashB)} km/h`);
}
const n = +(process.argv[2] || 3);
series('ren', 'rookie', n);
series('takeru', 'mio', n);
series('jin', 'yu', n);
series('rookie', 'takeru', n);
