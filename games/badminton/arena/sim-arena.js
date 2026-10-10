// CPU 同士で試合を回してバランスを見る:  node games/badminton/arena/sim-arena.js [試合数]
import { Match } from './match.js';
import { mulberry32 } from './physics.js';

function play(levelP, levelO, seed) {
  const m = new Match({ human: false, level: levelO, levelP, rng: mulberry32(seed) });
  let guard = 0;
  while (m.phase !== 'over' && guard++ < 400000) {
    m.step(1 / 60);
    if (m.phase === 'dead' && m.phaseT > 0.8) m.nextRally();
  }
  return m;
}

const n = +(process.argv[2] || 6);
const agg = { names: {}, reasons: {}, rally: [], kmh: [], wins: 0, games: 0, ms: 0 };
for (let i = 0; i < n; i++) {
  const t = performance.now();
  const m = play('expert', 'expert', 100 + i);
  agg.ms += performance.now() - t; agg.games++;
  if (m.score.P > m.score.O) agg.wins++;
  for (const h of m.stats.hits) { agg.names[h.name] = (agg.names[h.name] || 0) + 1; if (h.name.includes('スマッシュ')) agg.kmh.push(h.kmh); }
  for (const p of m.stats.points) { const k = p.reason + ':' + p.shot; agg.reasons[k] = (agg.reasons[k] || 0) + 1; agg.rally.push(p.rally); }
}
const top = o => Object.entries(o).sort((a, b) => b[1] - a[1]).slice(0, 14).map(([k, v]) => `${k} ${v}`).join(' / ');
const avg = a => a.reduce((x, y) => x + y, 0) / (a.length || 1);
console.log(`同レベル ${agg.games}試合: P勝 ${agg.wins}, 平均ラリー ${avg(agg.rally).toFixed(1)}打, 1試合 ${(agg.ms / agg.games).toFixed(0)}ms`);
console.log('ショット:', top(agg.names));
console.log('決着:', top(agg.reasons));
console.log('スマッシュ平均', avg(agg.kmh).toFixed(0), 'km/h');
for (const [a, b] of [['pro', 'beginner'], ['expert', 'club'], ['beginner', 'pro']]) {
  let w = 0, pts = [0, 0];
  for (let i = 0; i < Math.max(2, n / 2); i++) { const m = play(a, b, 500 + i); if (m.score.P > m.score.O) w++; pts[0] += m.score.P; pts[1] += m.score.O; }
  console.log(`${a} vs ${b}: 勝 ${w}/${Math.max(2, n / 2)} 得点 ${pts}`);
}
