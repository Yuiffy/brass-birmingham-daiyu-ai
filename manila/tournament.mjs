import { createGame, clone, rng, wealth, capacity, apply, transition } from './engine.mjs';
import { STRATEGIES, chooseStrategy, cargoDice } from './strategies.mjs';

export function quantile(values, q) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b), index = (sorted.length - 1) * q;
  const low = Math.floor(index), fraction = index - low;
  return sorted[low] * (1 - fraction) + sorted[Math.min(low + 1, sorted.length - 1)] * fraction;
}
export function describe(values) {
  if (!values.length) return { n: 0 };
  const n = values.length, mean = values.reduce((a,b) => a+b, 0) / n;
  const sd = Math.sqrt(values.reduce((v, x) => v + (x - mean) ** 2, 0) / Math.max(1, n - 1));
  const bins = new Map();
  for (const v of values) { const start = Math.floor(v / 20) * 20; bins.set(start, (bins.get(start) || 0) + 1); }
  return { n, mean, min: Math.min(...values), max: Math.max(...values), sd, p05: quantile(values,.05), p25: quantile(values,.25), median: quantile(values,.5), p75: quantile(values,.75), p95: quantile(values,.95), histogram: [...bins].sort((a,b)=>a[0]-b[0]).map(([lower,count])=>({lower,upper:lower+20,count})) };
}

export function playMatch({ lineup, seed, rotation = 0, initialCaptain = 0, validate = false, searchOptions = {} }) {
  if (!lineup.every(id => STRATEGIES[id])) throw Error('参赛策略无效');
  const n = lineup.length, s = createGame(n, seed), shuffledHands = s.players.map(p => clone(p));
  // Rotate the SAME identities and hands together. The oldest/first bidder remains
  // physical seat zero, so a complete rotation block controls initial seat advantage.
  for (let seat = 0; seat < n; seat++) s.players[seat] = shuffledHands[(seat - rotation + n) % n];
  const seatStrategies = Array.from({length:n}, (_, seat) => lineup[(seat-rotation+n)%n]);
  s.captain = initialCaptain; s.actor = initialCaptain;
  const randoms = lineup.map((_, identity) => rng((seed ^ Math.imul(identity+1,0x45d9f3b)) >>> 0));
  const decisions = Array(n).fill(0), captainWins = Array(n).fill(0), bidSpent = Array(n).fill(0), placements = Array(n).fill(0), debtMax = Array(n).fill(0);
  let steps = 0;
  const started = performance.now();
  while (s.phase !== 'finished') {
    if (++steps > 6000 || s.voyage > 80) throw Error(`对局未结束：seed=${seed}, ${lineup.join('/')}`);
    const seat=s.actor, identity=(seat-rotation+n)%n, phase=s.phase;
    const action=chooseStrategy(s,seatStrategies[seat],randoms[identity],searchOptions);
    decisions[seat]++;
    if(action.type==='place')placements[seat]++;
    const dice=action.type==='roll'?cargoDice(s,seed ^ 0x9e3779b9):null;
    if(validate)apply(s,action,dice);else transition(s,action,dice);
    if(phase==='auction'&&s.phase==='setup'){captainWins[s.captain]++;bidSpent[s.captain]+=s.bidder===null?0:s.bid;}
    if(validate){
      for(const p of s.players){if(p.cash<0||!Number.isFinite(p.cash)||p.mortgages.some((m,g)=>m>p.shares[g]))throw Error('资产不变量失败');}
      for(let g=0;g<4;g++)if(s.supply[g]+s.players.reduce((v,p)=>v+p.shares[g],0)!==5)throw Error('股份不守恒');
    }
    s.players.forEach((p,i)=>debtMax[i]=Math.max(debtMax[i],p.mortgages.reduce((a,b)=>a+b,0)));
  }
  const scores=s.players.map((_,p)=>wealth(s,p)),max=Math.max(...scores),ties=scores.filter(v=>v===max).length;
  return {seed,rotation,players:n,lineup,seatStrategies,voyages:s.voyage,steps,elapsedMs:performance.now()-started,
    results:s.players.map((p,seat)=>({identity:(seat-rotation+n)%n,seat,strategy:seatStrategies[seat],score:scores[seat],cash:p.cash,stockValue:scores[seat]-p.cash+15*p.mortgages.reduce((a,b)=>a+b,0),mortgages:p.mortgages.reduce((a,b)=>a+b,0),maxMortgages:debtMax[seat],win:scores[seat]===max?1/ties:0,rank:1+scores.filter(v=>v>scores[seat]).length,captainWins:captainWins[seat],bidSpent:bidSpent[seat],placements:placements[seat],decisions:decisions[seat]})),market:s.market,
    auctionHistory:s.history.map(h=>({voyage:h.voyage,identity:(h.captain-rotation+n)%n,price:h.bid,wealth:h.wealth.map((_,i)=>h.wealth[(i+rotation)%n])}))};
}

export function buildSchedule({ seeds = 100, startSeed = 20261007, counts = [3,4,5], strategies = Object.keys(STRATEGIES), mode = 'focal' } = {}) {
  if(!Number.isInteger(seeds)||seeds<1||seeds>100000)throw Error('种子数量无效');
  if(!counts.every(n=>[3,4,5].includes(n))||!strategies.every(id=>STRATEGIES[id]))throw Error('赛程参数无效');
  const tasks=[];
  for(let i=0;i<seeds;i++)for(const n of counts){
    const seed=(startSeed+Math.imul(i+1,2654435761))>>>0;
    if(mode==='mixed'){
      // Every ordered distinct lineup: all 4-player permutations of four styles,
      // or a selected subset supplied by the caller. No incompatible pooling.
      const permutations=(prefix,remaining)=>{if(prefix.length===n){tasks.push({lineup:prefix,seed,rotation:0,mode});return;}for(const id of remaining)permutations([...prefix,id],remaining.filter(x=>x!==id));};
      permutations([],strategies);
    } else if(mode==='focal'){
      for(const strategy of strategies)for(let rotation=0;rotation<n;rotation++)tasks.push({lineup:[strategy,...Array(n-1).fill('balanced')],seed,rotation,mode,focal:strategy});
    } else if(mode==='self'){
      for(const strategy of strategies)for(let rotation=0;rotation<n;rotation++)tasks.push({lineup:Array(n).fill(strategy),seed,rotation,mode});
    } else throw Error('未知赛程模式');
  }
  return tasks.map((t,id)=>({...t,id}));
}

export function aggregateMatches(matches, mode = 'focal') {
  const groups=new Map();
  for(const match of matches)for(const result of match.results){
    if(mode==='focal'&&result.identity!==0)continue;
    const key=`${match.players}/${result.strategy}`;
    if(!groups.has(key))groups.set(key,[]);
    groups.get(key).push({...result,seed:match.seed,rotation:match.rotation,voyages:match.voyages});
  }
  return [...groups].map(([key,rows])=>{
    const [count,strategy]=key.split('/'),blocks=new Map();
    for(const row of rows){if(!blocks.has(row.seed))blocks.set(row.seed,[]);blocks.get(row.seed).push(row);}
    const scoreBlocks=[...blocks.values()].map(rs=>rs.reduce((v,r)=>v+r.score,0)/rs.length);
    const winBlocks=[...blocks.values()].map(rs=>rs.reduce((v,r)=>v+r.win,0)/rs.length);
    const scoreStats=describe(rows.map(r=>r.score)),scoreSe=describe(scoreBlocks).sd/Math.sqrt(blocks.size),winRate=rows.reduce((v,r)=>v+r.win,0)/rows.length,winSe=describe(winBlocks).sd/Math.sqrt(blocks.size);
    return {players:Number(count),strategy,name:STRATEGIES[strategy].name,...scoreStats,seedBlocks:blocks.size,mean95:[scoreStats.mean-1.96*scoreSe,scoreStats.mean+1.96*scoreSe],winRate,win95:[Math.max(0,winRate-1.96*winSe),Math.min(1,winRate+1.96*winSe)],averageRank:rows.reduce((v,r)=>v+r.rank,0)/rows.length,averageVoyages:rows.reduce((v,r)=>v+r.voyages,0)/rows.length,averageCaptainWins:rows.reduce((v,r)=>v+r.captainWins,0)/rows.length,averageBidSpent:rows.reduce((v,r)=>v+r.bidSpent,0)/rows.length,averageMortgages:rows.reduce((v,r)=>v+r.mortgages,0)/rows.length,bySeat:Array.from({length:Number(count)},(_,seat)=>{const rs=rows.filter(r=>r.seat===seat);return{seat,n:rs.length,mean:rs.length?rs.reduce((v,r)=>v+r.score,0)/rs.length:null,winRate:rs.length?rs.reduce((v,r)=>v+r.win,0)/rs.length:null};}),blocks:[...blocks].map(([seed,rs])=>({seed,mean:rs.reduce((v,r)=>v+r.score,0)/rs.length,win:rs.reduce((v,r)=>v+r.win,0)/rs.length}))};
  }).sort((a,b)=>a.players-b.players||b.mean-a.mean);
}

export function pairedVsBalanced(summary) {
  return summary.filter(r=>r.strategy!=='balanced').map(row=>{
    const baseline=summary.find(r=>r.players===row.players&&r.strategy==='balanced');
    if(!baseline)return null;
    const base=new Map(baseline.blocks.map(b=>[b.seed,b]));
    const deltas=row.blocks.filter(b=>base.has(b.seed)).map(b=>b.mean-base.get(b.seed).mean);
    const stats=describe(deltas),se=stats.sd/Math.sqrt(deltas.length);
    return{players:row.players,strategy:row.strategy,n:stats.n,scoreDelta:stats.mean,mean95:[stats.mean-1.96*se,stats.mean+1.96*se]};
  }).filter(Boolean);
}
