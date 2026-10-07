import { PRICES, clone, rng, capacity, wealth, transition, determinize } from './engine.mjs';
import { policy, rankActions, reserveBid } from './ai.mjs';

// Presets are immutable experiment inputs. They change ONLY auction decisions;
// stock purchase, cargo setup, and placement keep the historical heuristic.
export const AUCTION_FORMULAS = {};
const add = (key, config) => { AUCTION_FORMULAS['formula-' + key] = Object.freeze(config); };
add('legacy', { kind: 'legacy', name: '原公式+3（等价对照）' });
for (const cap of [10, 14, 18, 22, 26, 30]) add('fixed' + cap, { kind: 'fixed', cap, buffer: 6, name: `固定上限${cap}` });
for (const base of [4, 7, 10]) for (const optionWeight of [.35, .7, 1]) for (const controlWeight of [0, .25, .75]) {
  add(`b${base}-o${optionWeight * 100}-c${controlWeight * 100}`, {
    kind: 'linear', base, optionWeight, controlWeight, horizon: .8, buffer: 6,
    name: `线性 b${base}/股${optionWeight}/控${controlWeight}`,
  });
}
for (const horizon of [.4, 1.2]) add('h' + horizon * 100, {
  kind: 'linear', base: 7, optionWeight: .7, controlWeight: .25, horizon, buffer: 6,
  name: `剩余涨幅系数${horizon}`,
});
for (const buffer of [0, 12]) add('buffer' + buffer, {
  kind: 'linear', base: 7, optionWeight: .7, controlWeight: .25, horizon: .8, buffer,
  name: `线性参考/预留${buffer}`,
});
for (const base of [4, 7, 10]) add('relative' + base, {
  kind: 'relative', base, optionWeight: .7, controlWeight: .75, horizon: .8, buffer: 6,
  name: `公开持股优势 b${base}`,
});
for (const horizon of [1, 0]) for (const objective of ['wealth', 'relative']) {
  add(`mc${horizon ? 'round' : 'full'}-${objective}`, {
    kind: 'mc', horizon, objective, samples: 8, buffer: 6, quote: 'two paired evaluations; local unit-cost correction',
    name: `${horizon ? '单航次' : '终局'}模拟/${objective === 'wealth' ? '财富' : '领先'}`,
  });
}

export function formulaFeatures(s, observer, horizon = .8) {
  const left = 5 - Math.max(...s.market), steps = Math.max(1, Math.round(left * horizon));
  const option = Math.max(0, ...s.supply.map((n, g) => n ? PRICES[Math.min(5, s.market[g] + steps)] - Math.max(5, PRICES[s.market[g]]) : 0));
  const lifts = s.market.map(level => PRICES[Math.min(5, level + 1)] - PRICES[level]);
  const weights = s.players[observer].shares.map((n, g) => n * lifts[g]);
  // Initial opponents' shares are private. Use public purchases and the observer's
  // own cards only; never use the true hidden holdings from the simulator.
  const advantages = weights.map((v, g) => v - Math.max(...s.players.filter((_, p) => p !== observer).map(p => p.bought[g] * lifts[g])));
  return { option, spread: Math.max(...weights) - Math.min(...weights), relativeSpread: Math.max(...advantages) - Math.min(...advantages) };
}

function leaf(s, observer, objective) {
  const values = s.players.map((p, i) => {
    const future = s.phase === 'finished' ? 0 : .4 * p.shares.reduce((v, n, g) => v + n * (Math.min(30, PRICES[s.market[g]] + 12) - PRICES[s.market[g]]), 0);
    return wealth(s, i) + future;
  });
  return values[observer] - (objective === 'relative' ? Math.max(...values.filter((_, p) => p !== observer)) : 0);
}

const setupCache = new Map();
export function continuationPolicy(s, random) {
  if (s.phase !== 'setup') return policy(s, 'aggressive', random);
  const p = s.players[s.captain], available = capacity(p);
  // setupScore depends on cash only through these thresholds; setupActions on
  // capacity only through stock affordability. Preserve the exact ranked ties.
  const cashFlags = s.market.map(level => Number(p.cash < Math.max(5, PRICES[level]) + 8));
  const buyFlags = s.market.map((level, g) => Number(s.supply[g] > 0 && available >= Math.max(5, PRICES[level])));
  const key = [s.market.join(), p.shares.join(), cashFlags.join(), buyFlags.join()].join('/');
  let ties = setupCache.get(key);
  if (!ties) {
    const ranked = rankActions(s);
    ties = ranked.filter(x => Math.abs(x.score - ranked[0].score) < 1e-9).map(x => x.action);
    if (setupCache.size >= 5000) setupCache.clear();
    setupCache.set(key, ties);
  }
  return ties[Math.floor(random() * ties.length)];
}

function continueWorld(s, observer, sampleSeed, definition) {
  const t = clone(s), voyage = t.voyage, tieRandom = rng(sampleSeed ^ 0xa5a5a5a5);
  let steps = 0;
  while (t.phase !== 'finished' && (!definition.horizon || t.voyage === voyage)) {
    if (++steps > 6000 || t.voyage > voyage + 80) throw Error('Auction valuation rollout did not terminate');
    // Frozen continuation for ALL players. No recursive calls to auction models.
    const action = continuationPolicy(t, tieRandom);
    const goods = action.type === 'roll' ? t.boats.filter(b => !b.destination).map(b => b.good) : [];
    let k = 0;
    const rollNumber = t.rolls + 1, voyageNumber = t.voyage;
    const dice = action.type === 'roll' ? () => rng((sampleSeed ^ Math.imul(voyageNumber, 73856093) ^ Math.imul(rollNumber, 19349663) ^ Math.imul(goods[k++] + 1, 83492791)) >>> 0)() : null;
    transition(t, action, dice);
  }
  return leaf(t, observer, definition.objective);
}

const quotes = new WeakMap();
function simulationLimit(s, id, definition, random) {
  const observer = s.actor, key = `${s.voyage}/${observer}/${id}`;
  let cache = quotes.get(s);
  if (!cache) { cache = new Map(); quotes.set(s, cache); }
  if (cache.has(key)) return cache.get(key);
  const max = Math.max(0, Math.floor(capacity(s.players[observer]) - definition.buffer));
  const worlds = Array.from({ length: definition.samples }, () => {
    const seed = Math.floor(random() * 4294967296) >>> 0, world = determinize(s, observer, rng(seed));
    const declined = clone(world); transition(declined, { type: 'pass' });
    return { world, seed, baseline: continueWorld(declined, observer, seed, definition) };
  });
  const evaluate = amount => {
    let total = 0;
    for (const { world, seed, baseline } of worlds) {
      const win = clone(world); win.bid = amount; win.bidder = observer; win.actor = observer;
      win.passed = win.players.map((_, p) => p !== observer);
      transition(win, { type: 'pass' });
      total += continueWorld(win, observer, seed, definition) - baseline;
    }
    return total / worlds.length;
  };
  // Start near the observed auction region, then simulate the suggested price
  // again to account for financing / purchase constraints. Both corrections
  // assume a local cost slope of -1; this is a tested approximation, not an exact
  // break-even solver. Conditional gain is always compared with immediate exit.
  const clip = amount => Math.max(0, Math.min(max, Math.floor(amount)));
  const anchor = Math.min(12, max), proposed = clip(anchor + evaluate(anchor));
  const limit = clip(proposed + evaluate(proposed));
  cache.set(key, limit);
  return limit;
}

export function formulaLimit(s, id, random = () => .5) {
  const definition = AUCTION_FORMULAS[id];
  if (!definition) throw Error(`Unknown auction formula: ${id}`);
  const available = capacity(s.players[s.actor]);
  if (definition.kind === 'legacy') return Math.min(available, reserveBid(s, s.actor) + 3);
  const cap = Math.max(0, available - definition.buffer);
  if (definition.kind === 'fixed') return Math.min(cap, definition.cap);
  if (definition.kind === 'mc') return Math.min(cap, simulationLimit(s, id, definition, random));
  const features = formulaFeatures(s, s.actor, definition.horizon);
  const control = definition.kind === 'relative' ? features.relativeSpread : features.spread;
  return Math.max(0, Math.min(cap, Math.floor(definition.base + definition.optionWeight * features.option + definition.controlWeight * control)));
}
