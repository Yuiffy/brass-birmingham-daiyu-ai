import { aggregateMatches, describe } from './tournament.mjs';

export const seedAt=(base,i)=>(base+Math.imul(i+1,2654435761))>>>0;
const rotations=(lineup,seed,metadata)=>Array.from({length:lineup.length},(_,rotation)=>({lineup,seed,rotation,mode:'league',...metadata}));

export function pairLeague(strategies,seeds,base,stage) {
  const tasks=[];
  for(let i=0;i<seeds;i++)for(let a=0;a<strategies.length;a++)for(let b=a+1;b<strategies.length;b++){
    const x=strategies[a],y=strategies[b],seed=seedAt(base,i),meta={pair:[x,y],stage};
    tasks.push(...rotations([x,y,y,y],seed,{...meta,composition:'1:3'}));
    tasks.push(...rotations([x,x,y,y],seed,{...meta,composition:'2:2-adjacent'}));
    tasks.push(...rotations([x,y,x,y],seed,{...meta,composition:'2:2-alternate'}));
    tasks.push(...rotations([x,x,x,y],seed,{...meta,composition:'3:1'}));
  }
  return tasks;
}

export function challengeLeague(challengers,opponents,seeds,base,stage) {
  return Array.from({length:seeds},(_,i)=>challengers.flatMap(challenger=>rotations([challenger,...opponents],seedAt(base,i),{stage,challenger}))).flat();
}

export function finalLeague(strategies,seeds,base,stage) {
  if(strategies.length!==4)throw Error('Finals require four strategies');
  // Every identity retains its hand and occupies every physical seat twice.
  return Array.from({length:seeds},(_,i)=>[[0,1,2,3],[0,2,1,3]].flatMap(order=>Array.from({length:4},(_,r)=>({lineup:strategies,seed:seedAt(base,i),seatOrder:order.map((_,seat)=>order[(seat-r+4)%4]),mode:'league',stage})))).flat();
}

export function ranked(summary) {return [...summary].sort((a,b)=>b.winRate-a.winRate||a.averageRank-b.averageRank||b.mean-a.mean);}

function groupedInterval(rows,key) {
  const blocks=new Map();
  for(const r of rows){if(!blocks.has(r.seed))blocks.set(r.seed,[]);blocks.get(r.seed).push(r[key]);}
  const means=[...blocks.values()].map(v=>v.reduce((a,b)=>a+b,0)/v.length),stats=describe(means);
  return {mean:stats.mean,seedBlocks:means.length,ci95:means.length>1?[stats.mean-1.96*stats.sd/Math.sqrt(means.length),stats.mean+1.96*stats.sd/Math.sqrt(means.length)]:null};
}

export function pairDetails(matches) {
  const pairs=new Map();
  for(const m of matches){if(!m.pair)continue;const key=m.pair.join('/');if(!pairs.has(key))pairs.set(key,[]);pairs.get(key).push(m);}
  return [...pairs.values()].map(ms=>{
    const [a,b]=ms[0].pair,summary=ranked(aggregateMatches(ms,'league'));
    const invasions=[a,b].map(challenger=>{
      const trials=ms.filter(m=>m.lineup.filter(id=>id===challenger).length===1);
      const rows=trials.map(m=>{const own=m.results.find(r=>r.strategy===challenger),others=m.results.filter(r=>r.strategy!==challenger);return {seed:m.seed,win:own.win,score:own.score,gap:own.score-others.reduce((v,r)=>v+r.score,0)/others.length};});
      return {challenger,resident:challenger===a?b:a,games:rows.length,win:groupedInterval(rows,'win'),score:groupedInterval(rows,'score'),gap:groupedInterval(rows,'gap')};
    });
    return {a,b,games:ms.length,summary,invasions};
  });
}

export function challengeDetails(matches) {return ranked(aggregateMatches(matches.map(m=>({...m,results:m.results.filter(r=>r.identity===0)})),'league'));}

export function pairedFinals(matches,strategies) {
  return strategies.flatMap((a,i)=>strategies.slice(i+1).map(b=>{
    const rows=matches.map(m=>({seed:m.seed,score:m.results.find(r=>r.strategy===a).score-m.results.find(r=>r.strategy===b).score,win:m.results.find(r=>r.strategy===a).win-m.results.find(r=>r.strategy===b).win}));
    return {a,b,score:groupedInterval(rows,'score'),win:groupedInterval(rows,'win')};
  }));
}
