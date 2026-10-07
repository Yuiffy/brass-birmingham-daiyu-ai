// Original implementation of the Zoch base rules. All randomness is explicit.
export const GOODS = [
  { name: '肉豆蔻', color: '#a76842', reward: 24, costs: [2, 3, 4] },
  { name: '丝绸', color: '#81678b', reward: 30, costs: [3, 4, 5] },
  { name: '人参', color: '#547f69', reward: 18, costs: [1, 2, 3] },
  { name: '玉石', color: '#5d8b96', reward: 36, costs: [3, 4, 5, 5] },
];
export const PRICES = [0, 5, 10, 15, 20, 30];
export const FEES = [4, 3, 2], PAYOUTS = [6, 8, 15];
export const clone = s => structuredClone(s);
export function rng(seed = 1) {
  let x = seed >>> 0;
  return () => { x += 0x6d2b79f5; let t = Math.imul(x ^ x >>> 15, 1 | x); t ^= t + Math.imul(t ^ t >>> 7, 61 | t); return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}
export function shuffle(a, random) { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; }
export const price = (s, g) => PRICES[s.market[g]];
export const wealth = (s, p) => s.players[p].cash + s.players[p].shares.reduce((v, n, g) => v + n * price(s, g), 0) - 15 * s.players[p].mortgages.reduce((a, b) => a + b, 0);
export const capacity = p => p.cash + 12 * p.shares.reduce((v, n, g) => v + n - p.mortgages[g], 0);
export function createGame(count = 4, seed = 20261007) {
  if (![3, 4, 5].includes(count)) throw Error('只支持 3–5 人');
  const random = rng(seed), deck = shuffle(Array.from({ length: 12 }, (_, i) => i % 4), random);
  const players = Array.from({ length: count }, (_, i) => ({ name: i ? `商人 ${i}` : '你', cash: 30, shares: [0, 0, 0, 0], bought: [0, 0, 0, 0], mortgages: [0, 0, 0, 0] }));
  for (const p of players) { p.shares[deck.pop()]++; p.shares[deck.pop()]++; }
  const s = { version: 1, seed, diceIndex: 0, players, market: [0, 0, 0, 0], supply: [0, 1, 2, 3].map(g => 5 - players.reduce((v, p) => v + p.shares[g], 0)), captain: 0, voyage: 0, log: [], history: [] };
  nextVoyage(s); return s;
}
function note(s, text) { s.log.push(text); if (s.log.length > 70) s.log.shift(); }
export function nextVoyage(s) {
  s.voyage++; s.phase = 'auction'; s.actor = s.captain; s.bid = 0; s.bidder = null;
  s.passed = s.players.map(() => false); s.stopped = s.players.map(() => false);
  s.remaining = s.players.map(() => s.players.length === 3 ? 4 : 3);
  s.boats = []; s.port = [null, null, null]; s.yard = [null, null, null]; s.pirates = [null, null]; s.pilots = [null, null]; s.insurer = null;
  s.wave = 0; s.turnInWave = 0; s.rolls = 0; s.boardQueue = []; s.routeQueue = [];
  s.lastDice = []; s.boardCurrent = null;
  note(s, `第 ${s.voyage} 航次 · 竞拍港口船长`);
}
function pay(s, p, amount, mandatory = false) {
  const player = s.players[p];
  if (!mandatory && capacity(player) < amount) throw Error('支付能力不足');
  while (player.cash < amount) {
    const g = [0, 1, 2, 3].filter(g => player.shares[g] > player.mortgages[g]).sort((a, b) => price(s, a) - price(s, b))[0];
    if (g === undefined) break;
    player.mortgages[g]++; player.cash += 12;
  }
  player.cash -= Math.min(player.cash, amount);
}
export function startsList() { const out = []; for (let a = 0; a <= 5; a++) for (let b = 0; b <= 5; b++) { const c = 9 - a - b; if (c >= 0 && c <= 5) out.push([a, b, c]); } return out; }
const STARTS = startsList();
export function setupActions(s) {
  const out = [], p = s.players[s.captain];
  for (let omit = 0; omit < 4; omit++) {
    const goods = [0, 1, 2, 3].filter(g => g !== omit);
    for (const buy of [-1, 0, 1, 2, 3]) {
      if (buy >= 0 && (!s.supply[buy] || capacity(p) < Math.max(5, price(s, buy)))) continue;
      for (const starts of STARTS) out.push({ type: 'setup', buy, goods, starts });
    }
  } return out;
}
export function rawPlacements(s) {
  const out = [];
  for (let i = 0; i < s.boats.length; i++) {
    const b = s.boats[i], k = b.crew.length;
    if (!b.destination && k < GOODS[b.good].costs.length) out.push({ type: 'place', where: 'boat', index: i, cost: GOODS[b.good].costs[k] });
  }
  for (const where of ['port', 'yard']) for (let i = 0; i < 3; i++) if (s[where][i] === null) out.push({ type: 'place', where, index: i, cost: FEES[i] });
  const pirate = s.pirates.indexOf(null);
  if (pirate >= 0) out.push({ type: 'place', where: 'pirates', index: pirate, cost: 5 });
  for (let i = 0; i < 2; i++) if (s.pilots[i] === null) out.push({ type: 'place', where: 'pilots', index: i, cost: i ? 5 : 2 });
  if (s.insurer === null) out.push({ type: 'place', where: 'insurance', index: 0, cost: 0 });
  return out;
}
export function pilotActions(s, large = false) {
  const out = [{ type: 'pilot', moves: [] }], active = s.boats.map((b, i) => b.destination ? -1 : i).filter(i => i >= 0);
  for (const i of active) for (const d of large ? [-2, -1, 1, 2] : [-1, 1]) if (s.boats[i].pos + d >= 0) out.push({ type: 'pilot', moves: [[i, d]] });
  if (large) for (const i of active) for (const j of active.filter(j => j > i)) for (const d of [-1, 1]) for (const e of [-1, 1]) if (s.boats[i].pos + d >= 0 && s.boats[j].pos + e >= 0) out.push({ type: 'pilot', moves: [[i, d], [j, e]] });
  return out;
}
export function legalActions(s) {
  if (s.phase === 'finished') return [];
  if (s.phase === 'auction') {
    const out = [{ type: 'pass' }];
    for (let amount = s.bid + 1; amount <= Math.floor(capacity(s.players[s.actor])); amount++) out.push({ type: 'bid', amount });
    return out;
  }
  if (s.phase === 'setup') return setupActions(s);
  if (s.phase === 'placement') {
    if (s.stopped[s.actor] || !s.remaining[s.actor]) return [{ type: 'skip' }];
    const raw = rawPlacements(s), p = s.players[s.actor], positive = raw.filter(a => a.cost > 0);
    const blind = !p.shares.some((n, g) => n > p.mortgages[g]) && p.cash < Math.min(...positive.map(a => a.cost));
    return [...raw.filter(a => blind ? a.where !== 'insurance' : a.cost <= capacity(p)).map(a => blind ? { ...a, cost: p.cash, blind: true } : a), { type: 'stop' }];
  }
  if (s.phase === 'roll') return [{ type: 'roll' }];
  if (s.phase === 'boarding') return [{ type: 'board', index: -1 }, ...s.boats.map((b, i) => !b.destination && b.pos === 13 && b.crew.length < GOODS[b.good].costs.length ? { type: 'board', index: i } : null).filter(Boolean)];
  if (s.phase === 'pilot-small' || s.phase === 'pilot-large') return pilotActions(s, s.phase === 'pilot-large');
  if (s.phase === 'pirate-route') return ['port', 'yard'].map(destination => ({ type: 'route', destination }));
  if (s.phase === 'settlement') return [{ type: 'settle' }];
  throw Error(`未知阶段 ${s.phase}`);
}
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
export function apply(s, action, random = null) {
  // Legal lookup also normalizes property order and prevents trusting UI-supplied costs.
  const found = legalActions(s).find(a => Object.keys(a).every(k => same(a[k], action[k])));
  if (!found) throw Error('非法行动');
  transition(s, found, random); return s;
}
// Internal entrypoint for already generated legal actions in rollouts.
export function transition(s, a, random = null) {
  const p = s.actor;
  if (a.type === 'bid' || a.type === 'pass') {
    if (a.type === 'bid') { s.bid = a.amount; s.bidder = p; note(s, `${s.players[p].name} 出价 ${a.amount}`); }
    else { s.passed[p] = true; note(s, `${s.players[p].name} 放弃竞拍`); }
    const active = s.passed.map((v, i) => v ? -1 : i).filter(i => i >= 0);
    if ((s.bidder !== null && active.length === 1) || !active.length) {
      if (s.bidder !== null) { s.captain = s.bidder; pay(s, s.captain, s.bid); }
      s.actor = s.captain; s.phase = 'setup'; note(s, `${s.players[s.captain].name} 以 ${s.bidder === null ? 0 : s.bid} 获得船长`);
    } else { do { s.actor = (s.actor + 1) % s.players.length; } while (s.passed[s.actor]); }
  } else if (a.type === 'setup') {
    if (a.buy >= 0) { pay(s, p, Math.max(5, price(s, a.buy))); s.players[p].shares[a.buy]++; s.players[p].bought[a.buy]++; s.supply[a.buy]--; }
    s.boats = a.goods.map((good, i) => ({ good, pos: a.starts[i], start: a.starts[i], crew: [], destination: null, plundered: false }));
    s.phase = 'placement'; s.actor = s.captain;
    note(s, `船长装载 ${a.goods.map(g => GOODS[g].name).join('、')}；起点 ${a.starts.join('/')}${a.buy >= 0 ? `；购入${GOODS[a.buy].name}股票` : ''}`);
  } else if (['place', 'skip', 'stop'].includes(a.type)) {
    if (a.type === 'stop') s.stopped[p] = true;
    if (a.type === 'place') {
      pay(s, p, a.cost); s.remaining[p]--;
      if (a.where === 'boat') s.boats[a.index].crew.push(p);
      else if (a.where === 'insurance') { s.insurer = p; s.players[p].cash += 10; }
      else s[a.where][a.index] = p;
      note(s, `${s.players[p].name} · ${actionLabel(s, a)}`);
    }
    s.turnInWave++;
    if (s.turnInWave < s.players.length) s.actor = (s.actor + 1) % s.players.length;
    else {
      s.turnInWave = 0; s.wave++;
      if (s.players.length === 3 && s.wave === 1) { s.actor = s.captain; }
      else if (s.rolls === 2) beginPilots(s);
      else { s.phase = 'roll'; s.actor = s.captain; }
    }
  } else if (a.type === 'roll') {
    const rand = random ?? rng((s.seed + Math.imul(++s.diceIndex, 0x9e3779b9)) >>> 0);
    s.rolls++; s.lastDice = s.boats.map(b => b.destination ? 0 : 1 + Math.floor(rand() * 6));
    for (let i = 0; i < s.boats.length; i++) { const b = s.boats[i]; if (!b.destination) { b.pos += s.lastDice[i]; if (b.pos > 13) b.destination = 'port'; } }
    note(s, `第 ${s.rolls} 次掷骰：${s.lastDice.join(' / ')}`);
    if (s.rolls === 2 && s.pirates.some(p => p !== null) && s.boats.some(b => !b.destination && b.pos === 13)) {
      s.boardQueue = s.pirates.map((p, slot) => p === null ? null : { p, slot }).filter(Boolean); nextBoard(s);
    } else if (s.rolls === 3) finishMovement(s);
    else { s.phase = 'placement'; s.actor = s.captain; }
  } else if (a.type === 'board') {
    if (a.index >= 0) { s.boats[a.index].crew.push(p); s.pirates[s.boardCurrent.slot] = null; }
    nextBoard(s);
  } else if (a.type === 'pilot') {
    for (const [i, d] of a.moves) { const b = s.boats[i]; b.pos += d; if (b.pos > 13) b.destination = 'port'; }
    if (s.phase === 'pilot-small') { if (s.pilots[1] !== null) { s.phase = 'pilot-large'; s.actor = s.pilots[1]; } else { s.phase = 'roll'; s.actor = s.captain; } }
    else { s.phase = 'roll'; s.actor = s.captain; }
    note(s, `${s.players[p].name} 领航：${a.moves.length ? a.moves.map(([i, d]) => `${GOODS[s.boats[i].good].name}${d > 0 ? '+' : ''}${d}`).join('、') : '不移动'}`);
  } else if (a.type === 'route') {
    const b = s.boats[s.routeQueue.shift()]; b.destination = a.destination;
    if (!s.routeQueue.length) { s.phase = 'settlement'; s.actor = s.captain; }
  } else if (a.type === 'settle') settle(s);
  else throw Error(`未实现 ${a.type}`);
}
function nextBoard(s) {
  if (s.boardQueue.length) { s.boardCurrent = s.boardQueue.shift(); s.actor = s.boardCurrent.p; s.phase = 'boarding'; }
  else { const left = s.pirates.filter(p => p !== null); s.pirates = [left[0] ?? null, left[1] ?? null]; s.phase = 'placement'; s.actor = s.captain; }
}
function beginPilots(s) {
  if (s.pilots[0] !== null) { s.phase = 'pilot-small'; s.actor = s.pilots[0]; }
  else if (s.pilots[1] !== null) { s.phase = 'pilot-large'; s.actor = s.pilots[1]; }
  else { s.phase = 'roll'; s.actor = s.captain; }
}
function finishMovement(s) {
  const pirates = s.pirates.filter(p => p !== null);
  s.routeQueue = [];
  for (let i = 0; i < s.boats.length; i++) {
    const b = s.boats[i]; if (b.destination) continue;
    if (b.pos === 13 && pirates.length) { b.plundered = true; b.crew = []; s.routeQueue.push(i); }
    else b.destination = b.pos >= 13 ? 'port' : 'yard';
  }
  if (s.routeQueue.length) { s.phase = 'pirate-route'; s.actor = pirates[0]; }
  else { s.phase = 'settlement'; s.actor = s.captain; }
}
function settle(s) {
  const before = s.players.map((_, i) => wealth(s, i)), pirates = s.pirates.filter(p => p !== null);
  for (const b of s.boats) {
    const crew = b.plundered ? pirates : b.destination === 'port' ? b.crew : [];
    for (const p of crew) s.players[p].cash += Math.floor(GOODS[b.good].reward / crew.length);
  }
  const ports = s.boats.filter(b => b.destination === 'port').length, yards = 3 - ports;
  for (let i = 0; i < ports; i++) if (s.port[i] !== null) s.players[s.port[i]].cash += PAYOUTS[i];
  // Insurance pays itself nothing, and receives other voyage profits before repairs.
  for (let i = 0; i < yards; i++) if (s.yard[i] !== null && s.yard[i] !== s.insurer) s.players[s.yard[i]].cash += PAYOUTS[i];
  if (s.insurer !== null) for (let i = 0; i < yards; i++) if (s.yard[i] !== s.insurer) pay(s, s.insurer, PAYOUTS[i], true);
  for (const b of s.boats) if (b.destination === 'port') s.market[b.good] = Math.min(5, s.market[b.good] + 1);
  const entry = { voyage: s.voyage, captain: s.captain, bid: s.bidder === null ? 0 : s.bid, market: [...s.market], wealth: s.players.map((_, i) => wealth(s, i)), ports, goods: s.boats.map(b => ({ good: b.good, destination: b.destination, plundered: b.plundered })) };
  entry.deltaAtSettlement = entry.wealth.map((w, i) => w - before[i]); s.history.push(entry);
  note(s, `${ports} 船到港；${yards} 船维修 · 本航次结算完成`);
  if (s.market.some(m => m === 5)) { s.phase = 'finished'; s.actor = 0; note(s, '货价达到 30 · 游戏结束'); }
  else nextVoyage(s);
}
export function redeem(s, p, g) {
  if (!Number.isInteger(p) || !s.players[p] || !Number.isInteger(g) || g < 0 || g > 3) throw Error('非法赎回');
  if (!s.players[p].mortgages[g] || s.players[p].cash < 15) throw Error('赎回需 15 比索现金');
  s.players[p].cash -= 15; s.players[p].mortgages[g]--; note(s, `${s.players[p].name} 赎回${GOODS[g].name}股票`);
}
export function actionLabel(s, a) {
  if (a.type === 'place') {
    const label = a.where === 'boat' ? `${GOODS[s.boats[a.index].good].name}船` : a.where === 'port' ? `港口 ${'ABC'[a.index]}` : a.where === 'yard' ? `船厂 ${'ABC'[a.index]}` : a.where === 'pilots' ? `${a.index ? '大' : '小'}领航员` : a.where === 'pirates' ? `海盗${a.index ? '船员' : '船长'}` : '保险员（领取 10）';
    return `${label} · ${a.cost} 比索${a.blind ? '（免费乘客）' : ''}`;
  }
  if (a.type === 'bid') return `出价 ${a.amount}`;
  if (a.type === 'pass') return '放弃竞拍';
  if (a.type === 'stop') return '停止本航次部署';
  if (a.type === 'skip') return '跳过（已停止部署）';
  if (a.type === 'roll') return '掷骰 · 船只前进';
  if (a.type === 'settle') return '结算本航次';
  if (a.type === 'board') return a.index < 0 ? '留在海盗船' : `登上${GOODS[s.boats[a.index].good].name}船`;
  if (a.type === 'route') return `劫掠后送往${a.destination === 'port' ? '港口' : '船厂'}`;
  if (a.type === 'pilot') return a.moves.length ? a.moves.map(([i, d]) => `${GOODS[s.boats[i].good].name} ${d > 0 ? '+' : ''}${d}`).join(' / ') : '不使用领航';
  return '确认装载与起点';
}
export function observation(s, observer) {
  const out = clone(s);
  out.players = out.players.map((p, i) => i === observer || s.phase === 'finished' ? p : { ...p, shares: null, mortgages: null, mortgageCount: p.mortgages.reduce((a, b) => a + b, 0), shareCount: p.shares.reduce((a, b) => a + b, 0) });
  // Public ledger must not expose private share-derived valuations before game end.
  if (s.phase !== 'finished') out.history = out.history.map(({ wealth, deltaAtSettlement, ...h }) => h);
  delete out.seed; delete out.diceIndex;
  return out;
}
export function determinize(s, observer, random) {
  const out = clone(s), pool = [];
  for (let g = 0; g < 4; g++) {
    const unknown = 5 - s.supply[g] - s.players.reduce((n, p) => n + p.bought[g], 0) - (s.players[observer].shares[g] - s.players[observer].bought[g]);
    for (let j = 0; j < unknown; j++) pool.push(g);
  }
  shuffle(pool, random);
  for (let i = 0; i < s.players.length; i++) if (i !== observer) {
    const p = out.players[i], count = p.mortgages.reduce((a, b) => a + b, 0);
    p.shares = [...p.bought]; p.shares[pool.pop()]++; p.shares[pool.pop()]++;
    p.mortgages = [0, 0, 0, 0];
    const cards = shuffle(p.shares.flatMap((n, g) => Array(n).fill(g)), random);
    for (let j = 0; j < count; j++) p.mortgages[cards[j]]++;
  }
  out.log = []; out.history = []; return out;
}
