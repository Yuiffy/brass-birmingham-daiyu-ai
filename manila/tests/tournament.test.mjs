import test from 'node:test';
import assert from 'node:assert/strict';
import { createGame, clone, rng, apply, determinize, observation, legalActions } from '../engine.mjs';
import { chooseStrategy, STRATEGIES, cargoDice } from '../strategies.mjs';
import { describe, buildSchedule, playMatch, aggregateMatches, pairedVsBalanced } from '../tournament.mjs';

test('distribution summaries interpolate percentiles and preserve histogram mass',()=>{
  const s=describe([0,10,20,30,40]);assert.equal(s.mean,20);assert.equal(s.min,0);assert.equal(s.max,40);assert.equal(s.median,20);assert.equal(s.p25,10);assert.equal(s.p95,38);assert.equal(s.histogram.reduce((v,b)=>v+b.count,0),5);assert.equal(describe([-10,0]).histogram[0].lower,-20);
});
test('focal schedule is a full rotation block for each seed/strategy/count',()=>{
  const tasks=buildSchedule({seeds:2,counts:[3,4,5],strategies:['balanced','greedy']});assert.equal(tasks.length,48);
  for(const n of [3,4,5])for(const strategy of ['balanced','greedy']){const rs=tasks.filter(t=>t.lineup.length===n&&t.focal===strategy);assert.equal(rs.length,2*n);for(const seed of new Set(rs.map(t=>t.seed)))assert.deepEqual(rs.filter(t=>t.seed===seed).map(t=>t.rotation),Array.from({length:n},(_,i)=>i));}
  assert.equal(buildSchedule({seeds:1,counts:[4],strategies:['balanced','greedy','cautious','aggressive'],mode:'mixed'}).length,24);
});
test('cargo dice independent of lane order and inactive boat consumption',()=>{
  const s=createGame(),t=clone(s);s.boats=[{good:0},{good:2},{good:3}];t.boats=[{good:3},{good:0},{good:2}];s.rolls=t.rolls=1;
  const r=cargoDice(s,123),q=cargoDice(t,123),a=[r(),r(),r()],b=[q(),q(),q()];assert.equal(a[0],b[1]);assert.equal(a[1],b[2]);assert.equal(a[2],b[0]);
});
test('real matches reproducible and each strategy finishes with all legal actions',()=>{
  for(const id of Object.keys(STRATEGIES)){
    const options={lineup:[id,'balanced','balanced'],seed:15,rotation:1,validate:true,searchOptions:{samples:2,width:3}};
    const a=playMatch(options),b=playMatch(options);delete a.elapsedMs;delete b.elapsedMs;assert.deepEqual(a,b);assert.ok(a.voyages>=5);assert.equal(a.results.reduce((v,r)=>v+r.win,0),1);assert.ok(a.results.every(r=>Number.isFinite(r.score)));assert.equal(a.results.find(r=>r.identity===0).seat,1);
  }
});
test('search uses observer information only and does not mutate live state',()=>{
  const a=createGame();for(let i=0;i<4;i++)apply(a,{type:'pass'});apply(a,{type:'setup',buy:-1,goods:[0,1,2],starts:[3,3,3]});
  const b=clone(a);[b.players[1].shares,b.players[2].shares]=[b.players[2].shares,b.players[1].shares];const before=clone(a);
  assert.deepEqual(observation(a,0),observation(b,0));assert.deepEqual(chooseStrategy(a,'search',rng(7),{samples:3,width:3}),chooseStrategy(b,'search',rng(7),{samples:3,width:3}));assert.deepEqual(a,before);
});
test('summary counts only focal identity, clusters rotations by seed, and pairs common seeds',()=>{
  const matches=[];
  for(const seed of [1,2])for(let rotation=0;rotation<3;rotation++)for(const strategy of ['balanced','greedy']){
    matches.push({seed,rotation,players:3,voyages:5,results:[{identity:0,seat:rotation,strategy,score:seed*10+(strategy==='greedy'?5:0),win:1,rank:1,captainWins:1,bidSpent:5,mortgages:0},{identity:1,seat:(rotation+1)%3,strategy:'balanced',score:999,win:0,rank:2,captainWins:0,bidSpent:0,mortgages:0}]});
  }
  const summary=aggregateMatches(matches);assert.ok(summary.every(r=>r.n===6&&r.seedBlocks===2));assert.equal(summary.find(r=>r.strategy==='balanced').mean,15);const paired=pairedVsBalanced(summary)[0];assert.equal(paired.scoreDelta,5);assert.deepEqual(paired.mean95,[5,5]);
});
