import { GOODS, PRICES, PAYOUTS, price, wealth, capacity, clone, rng, legalActions, setupActions, transition, determinize, observation, actionLabel } from './engine.mjs';

const probabilityCache = new Map();
export function diceProbability(position, rolls, threshold = 13) {
  const key = `${position}/${rolls}/${threshold}`;
  if (probabilityCache.has(key)) return probabilityCache.get(key);
  let counts = new Map([[position, 1]]);
  for (let r = 0; r < rolls; r++) { const next = new Map(); for (const [p, mass] of counts) for (let die = 1; die <= 6; die++) next.set(p + die, (next.get(p + die) || 0) + mass / 6); counts = next; }
  const result = [...counts].reduce((v, [p, mass]) => v + (p >= threshold ? mass : 0), 0);
  probabilityCache.set(key, result); return result;
}
function exactly(position, rolls, target) { return diceProbability(position, rolls, target) - diceProbability(position, rolls, target + 1); }
export function arrivalProbabilities(s) {
  const pirates = s.pirates.some(p => p !== null);
  return s.boats.map(b => b.destination ? Number(b.destination === 'port') : diceProbability(b.pos, 3 - s.rolls, pirates ? 14 : 13));
}
export function countProbabilities(probs) {
  let p = [1]; for (const chance of probs) { const next = Array(p.length + 1).fill(0); p.forEach((v, k) => { next[k] += v * (1 - chance); next[k + 1] += v * chance; }); p = next; } return p;
}
export function positionValue(s, observer) {
  const probs = arrivalProbabilities(s), counts = countProbabilities(probs), player = s.players[observer];
  let value = 0;
  for (let i = 0; i < s.boats.length; i++) {
    const b = s.boats[i], own = b.crew.filter(p => p === observer).length, n = b.crew.length;
    const lift = PRICES[Math.min(5, s.market[b.good] + 1)] - price(s, b.good);
    value += probs[i] * (own * GOODS[b.good].reward / Math.max(1, n) + lift * player.shares[b.good]);
    if (s.pirates.includes(observer)) value += exactly(b.pos, 3 - s.rolls, 13) * GOODS[b.good].reward * s.pirates.filter(p => p === observer).length / s.pirates.filter(p => p !== null).length;
  }
  for (let i = 0; i < 3; i++) {
    const portP = counts.reduce((v, p, k) => v + (k > i ? p : 0), 0), yardP = counts.reduce((v, p, k) => v + (3 - k > i ? p : 0), 0);
    if (s.port[i] === observer) value += PAYOUTS[i] * portP;
    if (s.yard[i] === observer && s.insurer !== observer) value += PAYOUTS[i] * yardP;
    if (s.insurer === observer && s.yard[i] !== observer) value -= PAYOUTS[i] * yardP;
  }
  return value;
}
export function placementScore(s, a) {
  if (a.type !== 'place') return 0;
  const probs = arrivalProbabilities(s), counts = countProbabilities(probs);
  if (a.where === 'boat') {
    const b = s.boats[a.index], own = b.crew.filter(p => p === s.actor).length;
    // Expected dilution is a baseline policy assumption, not an exact probability.
    const future = Math.max(0, (s.players.length === 3 ? 4 : 3) - s.wave - 1);
    const finalCrew = Math.min(GOODS[b.good].costs.length, b.crew.length + 1 + future * 0.55);
    const beforeCrew = Math.max(1, finalCrew - 1);
    return probs[a.index] * GOODS[b.good].reward * ((own + 1) / finalCrew - own / beforeCrew) - a.cost;
  }
  if (a.where === 'port' || a.where === 'yard') {
    const chance = counts.reduce((v, prob, k) => v + ((a.where === 'port' ? k : 3 - k) > a.index ? prob : 0), 0);
    return PAYOUTS[a.index] * chance - a.cost;
  }
  if (a.where === 'insurance') return 10 - counts.reduce((v, prob, k) => v + prob * PAYOUTS.slice(0, 3 - k).reduce((a, b) => a + b, 0), 0);
  if (a.where === 'pirates') {
    const alreadyOwn = s.pirates.filter(p => p === s.actor).length, pirates = s.pirates.filter(p => p !== null).length;
    let loot = s.boats.reduce((v, b) => v + (b.destination ? 0 : exactly(b.pos, 3 - s.rolls, 13) * GOODS[b.good].reward), 0);
    let score = loot * ((alreadyOwn + 1) / (pirates + 1) - alreadyOwn / Math.max(1, pirates)) - a.cost;
    if (s.rolls < 2) score += s.boats.reduce((v, b) => v + exactly(b.pos, 2 - s.rolls, 13) * GOODS[b.good].reward / (b.crew.length + 2), 0);
    return score;
  }
  if (a.where === 'pilots') {
    if (s.rolls !== 2) return -a.cost + 0.2 * s.players[s.actor].shares.reduce((a, b) => a + b, 0);
    const base = positionValue(s, s.actor), original = s.phase;
    const temp = clone(s); temp.phase = a.index ? 'pilot-large' : 'pilot-small';
    const best = Math.max(...legalActions(temp).map(pilot => { const t = clone(temp); transition(t, pilot); return positionValue(t, s.actor); }));
    return best - base - a.cost;
  }
  return -a.cost;
}
export function setupScore(s, a) {
  const p = s.players[s.captain]; let score = 0;
  for (let i = 0; i < 3; i++) {
    const g = a.goods[i], success = diceProbability(a.starts[i], 3);
    score += success * (p.shares[g] + Number(g === a.buy)) * (PRICES[Math.min(5, s.market[g] + 1)] - price(s, g));
    // First placement and diversification matter even when current shares are zero.
    score += 0.22 * Math.max(0, success * GOODS[g].reward / GOODS[g].costs.length - GOODS[g].costs[0]);
  }
  if (a.buy >= 0) {
    const cost = Math.max(5, price(s, a.buy));
    // Cheap stock option policy: only a heuristic for future liquidation.
    score += Math.min(30, price(s, a.buy) + 12) - cost;
    if (p.cash < cost + 8) score -= 4;
    if (!a.goods.includes(a.buy)) score -= 4;
  }
  return score;
}
export function reserveBid(s, p) {
  const own = s.players[p], left = 5 - Math.max(...s.market);
  // Finite-horizon stock option estimate. The captain can steer a bought cargo,
  // but a lagging cargo is unlikely to reach 30 before a market leader ends play.
  const option = Math.max(0, ...s.supply.map((n, g) => n ? PRICES[Math.min(5, s.market[g] + Math.max(1, Math.round(left * .8)))] - Math.max(5, price(s, g)) : 0));
  const weights = own.shares.map((n,g) => n * (PRICES[Math.min(5,s.market[g]+1)] - price(s,g)));
  const control = (Math.max(...weights) - Math.min(...weights)) * .25;
  return Math.max(0, Math.min(Math.floor(capacity(own)) - 6, Math.floor(4 + option * .7 + control)));
}
export function rankActions(s) {
  if (s.phase === 'setup') return setupActions(s).map(a => ({ action: a, score: setupScore(s, a) })).sort((a, b) => b.score - a.score);
  const actions = legalActions(s);
  return actions.map(a => {
    let score = 0;
    if (s.phase === 'placement') score = placementScore(s, a);
    else if (s.phase.startsWith('pilot-')) { const t = clone(s); transition(t, a); score = positionValue(t, s.actor); }
    else if (s.phase === 'boarding') {
      if (a.index >= 0) { const b = s.boats[a.index]; score = GOODS[b.good].reward / (b.crew.length + 1); }
      else score = s.boats.reduce((v, b) => v + (b.destination ? 0 : exactly(b.pos, 1, 13) * GOODS[b.good].reward), 0) / s.pirates.filter(p => p !== null).length;
    } else if (s.phase === 'pirate-route') {
      const t = clone(s); transition(t, a); score = positionValue(t, s.actor);
    } else if (s.phase === 'auction') score = a.type === 'bid' && a.amount === s.bid + 1 && a.amount <= reserveBid(s, s.actor) ? 1 : a.type === 'pass' ? 0 : -1;
    return { action: a, score };
  }).sort((a, b) => b.score - a.score);
}
export function policy(s, style = 'balanced', random = () => 0.5) {
  // Does not read other players' private shares. Seed only breaks equal policy scores.
  if (s.phase === 'auction') {
    const limit = reserveBid(s, s.actor) + (style === 'aggressive' ? 3 : style === 'cautious' ? -3 : 0);
    return s.bid + 1 <= limit && s.bid + 1 <= capacity(s.players[s.actor]) ? { type: 'bid', amount: s.bid + 1 } : { type: 'pass' };
  }
  const ranked = rankActions(s);
  if (style === 'random' && s.phase === 'placement') return ranked[Math.floor(random() * ranked.length)].action;
  const ties = ranked.filter(x => Math.abs(x.score - ranked[0].score) < 1e-9);
  return ties[Math.floor(random() * ties.length)].action;
}
export function advice(s) { return rankActions(s).slice(0, 5).map(x => ({ ...x, label: actionLabel(s, x.action) })); }

// Exact known dynamics behind a small world-model interface. Learned dynamics must
// be calibrated against this adapter before being used for financial valuations.
export const ruleWorldModel = {
  id: 'manila-zoch-exact-v1',
  observe: (s, observer) => observation(s, observer),
  step: (s, a, random) => { const next = clone(s), before = s.players.map((_, p) => wealth(s, p)); transition(next, a, random); return { next, rewards: next.players.map((_, p) => wealth(next, p) - before[p]), terminal: next.phase === 'finished' }; },
};

function stats(values) {
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance = values.reduce((v, x) => v + (x - mean) ** 2, 0) / Math.max(1, values.length - 1);
  return { mean, sd: Math.sqrt(variance), se: Math.sqrt(variance / values.length) };
}
function forcedWin(s, observer, amount) {
  const t = clone(s); t.phase = 'auction'; t.actor = observer; t.bid = amount; t.bidder = observer;
  t.passed = t.players.map((_, i) => i !== observer);
  // The auction resolver pays the conditional winning price, including forced loans.
  transition(t, { type: 'pass' }); return t;
}
function declined(s, observer) {
  const t = clone(s); t.actor = observer; transition(t, { type: 'pass' }); return t;
}
function rollout(s, seed, style, horizon, observer = null) {
  const t = clone(s), random = rng(seed), start = t.voyage;
  let steps = 0;
  // The same coordinate-indexed die stream is used for both counterfactual worlds.
  // Policy tie breaking uses a separate stream and cannot consume dice randomness.
  while (t.phase !== 'finished' && (horizon === 0 || t.voyage < start + horizon)) {
    if (++steps > 6000 || t.voyage > start + 80) throw Error('模拟未收敛');
    const action = policy(t, t.actor === observer ? 'balanced' : style, random);
    let dice;
    if (action.type === 'roll') {
      // Engine consumes only active boat dice; map each call to its actual cargo.
      const activeGoods = t.boats.filter(b => !b.destination).map(b => b.good); let k = 0;
      dice = () => rng((seed ^ Math.imul(t.voyage - start + 1, 73856093) ^ Math.imul(t.rolls, 19349663) ^ Math.imul(activeGoods[k++] + 1, 83492791)) >>> 0)();
    }
    transition(t, action, dice);
  }
  const values = t.players.map((_, i) => wealth(t, i)), max = Math.max(...values), ties = values.filter(w => w === max).length;
  return { values, wins: values.map(w => w === max ? 1 / ties : 0), terminal: t.phase === 'finished', voyages: t.voyage - start };
}
export async function analyzeAuction(s, observer = s.actor, options = {}, progress = () => {}) {
  if (s.phase !== 'auction' || observer !== s.actor) throw Error('请在自己的竞拍回合分析');
  const samples = Math.floor(Math.max(16, Math.min(1024, Number(options.samples) || 64)));
  const horizon = [0, 1, 3].includes(options.horizon) ? options.horizon : 1;
  const style = ['balanced', 'cautious', 'aggressive', 'random'].includes(options.style) ? options.style : 'balanced';
  const seed = Number(options.seed) || 20261007;
  const maxBid = Math.min(Math.floor(capacity(s.players[observer])), Math.max(s.bid + 1, Math.floor(Number(options.maxBid) || 25)));
  const candidates = Array.from({ length: maxBid - s.bid }, (_, i) => s.bid + i + 1);
  if (!candidates.length) return { rows: [], reason: '支付能力不足，无法加价', samples, horizon, style };
  const worlds = [], baseline = [];
  for (let i = 0; i < samples; i++) {
    const sampleSeed = (seed + Math.imul(i + 1, 2654435761)) >>> 0;
    const world = determinize(s, observer, rng(sampleSeed)); worlds.push({ world, seed: sampleSeed });
    baseline.push(rollout(declined(world, observer), sampleSeed, style, horizon, observer));
  }
  const rows = [];
  for (let j = 0; j < candidates.length; j++) {
    const bid = candidates[j], deltas = [], margins = [], winDeltas = []; let winRate = 0, terminal = 0;
    for (let i = 0; i < samples; i++) {
      const { world, seed } = worlds[i], result = rollout(forcedWin(world, observer, bid), seed, style, horizon, observer), base = baseline[i];
      deltas.push(result.values[observer] - base.values[observer]);
      margins.push(result.values[observer] - Math.max(...result.values.filter((_, p) => p !== observer)) - (base.values[observer] - Math.max(...base.values.filter((_, p) => p !== observer))));
      winRate += result.wins[observer] / samples; winDeltas.push(result.wins[observer] - base.wins[observer]); terminal += Number(result.terminal);
    }
    const st = stats(deltas), margin = stats(margins);
    rows.push({ bid, delta: st.mean, lower: st.mean - 1.96 * st.se, upper: st.mean + 1.96 * st.se, sd: st.sd, marginDelta: margin.mean, winRate: terminal === samples ? winRate : null, winDelta: terminal === samples ? stats(winDeltas).mean : null });
    progress({ completed: j + 1, total: candidates.length });
    await new Promise(resolve => setTimeout(resolve, 0));
  }
  const profitable = rows.filter(r => r.delta >= 0).map(r => r.bid), conservative = rows.filter(r => r.lower >= 0).map(r => r.bid);
  return { model: ruleWorldModel.id, samples, horizon, style, seed, rows, meanAcceptableBids: profitable, conservativeBids: conservative, breakEven: profitable.length ? Math.max(...profitable) : 0, conservativeMax: conservative.length ? Math.max(...conservative) : 0, rangeCensored: rows.at(-1).delta >= 0 && rows.at(-1).bid < Math.floor(capacity(s.players[observer])), baselineWealth: stats(baseline.map(r => r.values[observer])).mean, baselineWin: baseline.every(r => r.terminal) ? stats(baseline.map(r => r.wins[observer])).mean : null, assumption: '条件于以该价格赢得船长，对比现在永久退出竞拍；其他玩家按所选基线策略继续竞拍。区间为配对均值的近似 95% 蒙特卡洛置信区间，不包含策略偏差。短期模式只计当前市值。' };
}
export function simulateGame(state, seed = 1, style = 'balanced') {
  return rollout(state, seed, style, 0);
}
