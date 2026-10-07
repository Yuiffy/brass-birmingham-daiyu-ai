import { GOODS, PRICES, clone, rng, legalActions, transition, determinize, wealth, capacity } from './engine.mjs';
import { policy, rankActions, setupScore, positionValue, reserveBid } from './ai.mjs';

export const STRATEGIES = {
  balanced: { name: '均衡', description: '原版均衡启发式，包含未来船员分摊和股票期权估计。' },
  cautious: { name: '保守竞价', description: '与均衡相同，船长竞拍上限低 3 比索。' },
  aggressive: { name: '积极竞价', description: '与均衡相同，船长竞拍上限高 3 比索。' },
  greedy: { name: '即时收益', description: '部署按当前船员组成的即时期望净收益排序，不预测后续分摊；装载不计股票远期权利。竞价沿用均衡。' },
  random: { name: '随机部署', description: '部署阶段在合法动作中均匀随机，含停止；其余阶段沿用均衡。' },
  search: { name: '模型前瞻', description: '每次重采样隐藏股，最多 4 个候选 × 8 个样本模拟至本航次结算，使用股票远期启发式作叶估值；后续策略与竞价沿用均衡。不是 MuZero 或树搜索。' },
};

// Keep historical IDs intact. League variants are parsed in every worker, so
// arbitrary candidate offsets do not require mutation of a global registry.
export function strategyDefinition(id) {
  if (STRATEGIES[id]) return STRATEGIES[id];
  const match = /^(bid|look|relative)(-liquid)?\+(\d+)$/.exec(id);
  if (!match || Number(match[3]) > 96) return null;
  const kind=match[1], liquid=Boolean(match[2]), offset=Number(match[3]);
  return {kind,liquid,offset,name:`${kind==='bid'?'积极':kind==='look'?'前瞻财富':'前瞻领先'}${liquid?'保留资金':''}+${offset}`,description:`均衡竞价上限增加 ${offset}；${liquid?'支付能力至少保留 6 比索；':''}${kind==='bid'?'原部署启发式':`精确规则单航次前瞻，叶估值目标为${kind==='look'?'自身财富':'自身减最强对手财富'}，计算预算由比赛配置指定`}。`};
}

function auctionPolicy(s, definition) {
  const available=capacity(s.players[s.actor]);
  const limit=Math.min(available-(definition.liquid?6:0),reserveBid(s,s.actor)+definition.offset);
  return s.bid+1<=limit?{type:'bid',amount:s.bid+1}:{type:'pass'};
}

function rolloutPolicy(s, ids, random) {
  const id=ids?.[s.actor] || 'balanced', definition=strategyDefinition(id);
  if (definition?.kind && s.phase==='auction')return auctionPolicy(s,definition);
  // Model-based opponents use their heuristic continuation in these rollouts;
  // recursively nesting their planners would change the fixed compute budget.
  return policy(s, ['balanced','cautious','aggressive','random'].includes(id)?id:'balanced',random);
}

// A cargo-indexed exogenous tape keeps dice comparable when policies change order,
// skip finished boats, or reach different turn counts. No strategy sees this seed.
export function cargoDice(s, seed) {
  const goods = s.boats.filter(b => !b.destination).map(b => b.good);
  const voyage = s.voyage, roll = s.rolls + 1; let i = 0;
  return () => rng((seed ^ Math.imul(voyage, 73856093) ^ Math.imul(roll, 19349663) ^ Math.imul(goods[i++] + 1, 83492791)) >>> 0)();
}

function greedyPolicy(s, random) {
  if (!['placement', 'setup'].includes(s.phase)) return policy(s, 'balanced', random);
  const p = s.actor, before = positionValue(s, p), beforeWealth = wealth(s, p);
  const ranked = legalActions(s).map(action => {
    if (s.phase === 'setup') {
      let score = setupScore(s, action);
      if (action.buy >= 0) {
        const price = PRICES[s.market[action.buy]], cost = Math.max(5, price);
        // Remove the baseline's assumed future liquidation option.
        score -= Math.min(30, price + 12) - cost;
        score += price - cost;
      }
      return { action, score };
    }
    const t = clone(s); transition(t, action);
    return { action, score: wealth(t, p) - beforeWealth + positionValue(t, p) - before };
  }).sort((a, b) => b.score - a.score);
  const ties = ranked.filter(x => Math.abs(x.score - ranked[0].score) < 1e-9);
  return ties[Math.floor(random() * ties.length)].action;
}

function leafValue(s, p) {
  if (s.phase === 'finished') return wealth(s, p);
  const lead = Math.max(...s.market), remaining = 5 - lead;
  const option = s.players[p].shares.reduce((v, n, g) => v + n * (PRICES[Math.min(5, s.market[g] + Math.max(1, Math.round(remaining * .65)))] - PRICES[s.market[g]]), 0);
  return wealth(s, p) + .4 * option;
}

export function searchPolicy(s, random, { samples = 8, width = 4, objective = 'wealth', rolloutStrategies } = {}) {
  if (!['placement','setup','pilot-small','pilot-large','boarding','pirate-route'].includes(s.phase)) return policy(s, 'balanced', random);
  const ranked = rankActions(s);
  if (ranked.length === 1) return ranked[0].action;
  let candidates;
  if (s.phase === 'setup') {
    // Include the best setup from different stock/cargo choices, not four adjacent starts.
    const seen = new Set(); candidates = ranked.filter(({ action }) => {
      const key = `${action.buy}/${action.goods.join()}`;
      if (seen.has(key)) return false; seen.add(key); return true;
    }).slice(0, width);
  } else {
    candidates = ranked.slice(0, width);
    const stop = ranked.find(x => ['stop','skip'].includes(x.action.type));
    if (stop && !candidates.includes(stop)) candidates[candidates.length - 1] = stop;
  }
  const actor = s.actor, totals = candidates.map(() => 0), seeds = Array.from({ length: samples }, () => Math.floor(random() * 4294967296));
  for (const seed of seeds) {
    const world = determinize(s, actor, rng(seed));
    for (let c = 0; c < candidates.length; c++) {
      const t = clone(world), tieRandom = rng(seed ^ 0xa5a5a5a5), voyage = t.voyage;
      transition(t, candidates[c].action);
      let steps = 0;
      while (t.phase !== 'finished' && t.voyage === voyage) {
        if (++steps > 250) throw Error('前瞻模拟超出单航次动作上限');
        const action = rolloutPolicy(t, rolloutStrategies, tieRandom);
        transition(t, action, action.type === 'roll' ? cargoDice(t, seed) : null);
      }
      totals[c] += leafValue(t, actor) - (objective==='relative'?Math.max(...t.players.map((_,p)=>p===actor?-Infinity:leafValue(t,p))):0);
    }
  }
  return candidates[totals.indexOf(Math.max(...totals))].action;
}

export function chooseStrategy(s, id, random = () => .5, searchOptions = {}) {
  const definition=strategyDefinition(id);
  if (!definition) throw Error(`未知策略 ${id}`);
  if(definition.kind){
    if(s.phase==='auction')return auctionPolicy(s,definition);
    if(definition.kind!=='bid')return searchPolicy(s,random,{...searchOptions,objective:definition.kind==='relative'?'relative':'wealth'});
    return policy(s,'balanced',random);
  }
  if (id === 'greedy') return greedyPolicy(s, random);
  if (id === 'search') return searchPolicy(s, random, searchOptions);
  return policy(s, id, random);
}
