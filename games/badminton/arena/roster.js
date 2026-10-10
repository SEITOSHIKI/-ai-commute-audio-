/*
 * 選手名鑑と育成。
 * 実在の選手は名前・肖像の権利があるため収録しない。プレースタイルの「型」を元にした架空の選手。
 * 名前は端末内で自由に書き換えられる（共有・公開するビルドには含まれない）。
 *
 * 能力値（0〜99）: power パワー / speed スピード / defense 守備 / control コントロール /
 *                  technique テクニック / reaction 反応
 */
import { DEF_ATTR } from './physics.js';

export const STAT_KEYS = ['power', 'speed', 'defense', 'control', 'technique', 'reaction'];
export const STAT_JP = { power: 'パワー', speed: 'スピード', defense: '守備', control: 'コントロール', technique: 'テクニック', reaction: '反応' };
export const STAT_HINT = {
  power: 'スマッシュの初速とジャンプの高さ',
  speed: 'フットワークの速さと加速',
  defense: '速い球を受ける上手さとリーチ',
  control: '狙ったところに打てる精度',
  technique: 'カットの曲がりとスピンネットの回転、相手を惑わす力',
  reaction: '相手が打ってから動き出すまでの速さ',
};

// 体格: height(m) / build（0 細身〜1 がっしり）/ 利き手 / 見た目
export const CHARACTERS = [
  {
    id: 'ren', name: '神城 レン', style: 'レフティのオールラウンダー',
    note: '低い姿勢からのネット前と、崩れないレシーブ。ミスをしない配球で相手を追い込む。',
    height: 1.75, build: 0.45, lefty: true, skin: 0xe2b993, hair: 0x1b1714, hairStyle: 'short', shirt: 0xf4b63f, shorts: 0x1b2a33,
    stats: { power: 76, speed: 84, defense: 90, control: 92, technique: 90, reaction: 86 },
  },
  {
    id: 'jin', name: '鷹宮 ジン', style: '手首で騙すトリックスター',
    note: '同じ構えからスマッシュとカットを打ち分ける。高い打点からの急角度が武器。',
    height: 1.81, build: 0.4, lefty: false, skin: 0xe8c29c, hair: 0x2a211b, hairStyle: 'swept', shirt: 0x6fc3e8, shorts: 0x10202b,
    stats: { power: 88, speed: 80, defense: 72, control: 78, technique: 94, reaction: 82 },
  },
  {
    id: 'yu', name: '早瀬 ユウ', style: 'レフティの速攻・前衛型',
    note: '小柄で反応が速い。前に詰めてのプッシュとドライブ戦で主導権を握る。',
    height: 1.67, build: 0.35, lefty: true, skin: 0xe6be97, hair: 0x15110f, hairStyle: 'spiky', shirt: 0x5fd6a2, shorts: 0x14262a,
    stats: { power: 72, speed: 92, defense: 80, control: 80, technique: 88, reaction: 94 },
  },
  {
    id: 'takeru', name: '大河 タケル', style: '長身の剛腕スマッシャー',
    note: '190cm 近い打点から 350km/h 級を叩き込む。動かされると脆い。',
    height: 1.89, build: 0.8, lefty: false, skin: 0xd9ad85, hair: 0x221a15, hairStyle: 'buzz', shirt: 0xee6c5a, shorts: 0x22191a,
    stats: { power: 98, speed: 68, defense: 70, control: 70, technique: 64, reaction: 74 },
  },
  {
    id: 'mio', name: '白石 ミオ', style: '粘りのディフェンダー',
    note: 'どんな球も拾って長いラリーに持ち込む。相手のミスを待つ守備型。',
    height: 1.64, build: 0.3, lefty: false, skin: 0xf0cda8, hair: 0x2b1c14, hairStyle: 'pony', shirt: 0xc79bf0, shorts: 0x1f1a2b,
    stats: { power: 62, speed: 90, defense: 96, control: 86, technique: 78, reaction: 88 },
  },
  {
    id: 'rookie', name: 'ルーキー', style: '育成枠（あなた）',
    note: '伸びしろの塊。試合で経験値を積み、能力を自分で割り振って育てる。',
    height: 1.74, build: 0.5, lefty: false, skin: 0xe4bd96, hair: 0x241b16, hairStyle: 'short', shirt: 0xf1f4ec, shorts: 0x2a3a3d,
    stats: { power: 58, speed: 60, defense: 58, control: 60, technique: 56, reaction: 58 }, rookie: true,
  },
];

const KEY = 'arena.roster.v1';
const MAX_STAT = 99;

function load(store) {
  const d = store.get(KEY);
  return d && typeof d === 'object' ? d : {};
}

// 保存された育成状態を重ねた選手データ
export function getCharacter(id, store) {
  const base = CHARACTERS.find(c => c.id === id) || CHARACTERS[0];
  const saved = load(store)[base.id] || {};
  const stats = { ...base.stats };
  for (const k of STAT_KEYS) if (saved.stats && Number.isFinite(saved.stats[k])) stats[k] = Math.min(MAX_STAT, saved.stats[k]);
  return {
    ...base,
    name: typeof saved.name === 'string' && saved.name.trim() ? saved.name.trim().slice(0, 12) : base.name,
    stats,
    level: saved.level || 1,
    xp: saved.xp || 0,
    points: saved.points || 0,
    record: saved.record || { w: 0, l: 0 },
  };
}

function save(store, ch) {
  const all = load(store);
  all[ch.id] = { name: ch.name, stats: ch.stats, level: ch.level, xp: ch.xp, points: ch.points, record: ch.record };
  store.set(KEY, all);
}

export const xpToNext = level => 80 + level * 40;

/**
 * 試合結果から経験値を付与。レベルが上がるごとに能力ポイント3。
 * 戻り値: { gained, levelsUp, ch }
 */
export function awardMatch(store, id, { won, aces, hardShots, points }) {
  const ch = getCharacter(id, store);
  const gained = (won ? 90 : 45) + aces * 6 + hardShots * 2 + points * 2;
  ch.xp += gained;
  let levelsUp = 0;
  while (ch.xp >= xpToNext(ch.level)) { ch.xp -= xpToNext(ch.level); ch.level++; ch.points += 3; levelsUp++; }
  ch.record = { w: ch.record.w + (won ? 1 : 0), l: ch.record.l + (won ? 0 : 1) };
  save(store, ch);
  return { gained, levelsUp, ch };
}

// 能力ポイントを1振る（99 まで。高い能力ほど伸ばしにくい = 2ポイント必要）
export const costFor = v => (v >= 90 ? 2 : 1);
export function allocate(store, id, key) {
  const ch = getCharacter(id, store);
  const cost = costFor(ch.stats[key]);
  if (!STAT_KEYS.includes(key) || ch.points < cost || ch.stats[key] >= MAX_STAT) return ch;
  ch.stats[key] += 1; ch.points -= cost;
  save(store, ch);
  return ch;
}

export function rename(store, id, name) {
  const ch = getCharacter(id, store);
  ch.name = String(name || '').trim().slice(0, 12) || CHARACTERS.find(c => c.id === id).name;
  save(store, ch);
  return ch;
}

export function resetCharacter(store, id) {
  const all = load(store); delete all[id]; store.set(KEY, all);
}

/**
 * 能力値と体格から、物理エンジンで使う能力（attr）を作る。
 * 例: パワー 98 → スマッシュ最高初速 約 360km/h、スピード 92 → 約 5.1 m/s
 */
export function toAttr(ch) {
  const s = k => ch.stats[k] / 100;
  const h = ch.height;
  return {
    ...DEF_ATTR,
    speed: 3.75 + 1.45 * s('speed') - ch.build * 0.12,
    acc: 20 + 10 * s('speed'),
    reach: 0.6 * h + 0.12 * s('defense'),
    reachTop: 1.32 * h + 0.23,
    jumpV: 2.6 + 0.75 * s('power'),
    vOver: 66 + 30 * s('power') + ch.build * 4,
    vSide: 44 + 16 * s('power'),
    vUnder: 34 + 10 * s('power'),
    noise: 1.45 - 0.9 * s('control'),
    sliceK: 0.55 + 0.7 * s('technique'),
    deceive: 0.02 + 0.08 * s('technique'),
    defense: s('defense'),
    react: 1.35 - 0.55 * s('reaction'),
    lefty: !!ch.lefty,
  };
}

// 総合力（表示用）
export const overall = ch => Math.round(STAT_KEYS.reduce((a, k) => a + ch.stats[k], 0) / STAT_KEYS.length);
