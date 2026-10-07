import test from 'node:test';
import assert from 'node:assert/strict';
import { createGame, rng, capacity, clone, apply, observation } from '../engine.mjs';
import { reserveBid } from '../ai.mjs';
import { chooseStrategy, strategyDefinition } from '../strategies.mjs';
import { playMatch, aggregateMatches } from '../tournament.mjs';
import { pairLeague, finalLeague, challengeLeague, pairDetails, challengeDetails } from '../league.mjs';

test('high auction offsets respect actual credit and liquidity buffer',()=>{
  const s=createGame(4,10);s.bid=reserveBid(s,s.actor)+3;
  assert.equal(chooseStrategy(s,'bid+3').type,'pass');assert.equal(chooseStrategy(s,'bid+6').type,'bid');
  s.bid=capacity(s.players[s.actor])-1;
  assert.equal(chooseStrategy(s,'bid+60').type,'bid');assert.equal(chooseStrategy(s,'bid-liquid+60').type,'pass');
  s.bid++;assert.equal(chooseStrategy(s,'bid+60').type,'pass');
  assert.equal(strategyDefinition('bid+999'),null);assert.equal(strategyDefinition('relative-liquid+6').liquid,true);
});

test('pair schedules balance frequency, seats, and one-versus-three challenges',()=>{
  const tasks=pairLeague(['bid+3','bid+6','bid+24'],2,123,'test');assert.equal(tasks.length,96);
  const counts=Object.fromEntries(['bid+3','bid+6','bid+24'].map(id=>[id,tasks.flatMap(t=>t.lineup).filter(x=>x===id).length]));assert.equal(new Set(Object.values(counts)).size,1);
  for(const composition of ['1:3','2:2-adjacent','2:2-alternate','3:1'])assert.equal(tasks.filter(t=>t.composition===composition).length,24);
  const finals=finalLeague(['bid+3','bid+6','look+3','relative+6'],2,456,'final');assert.equal(finals.length,16);
  for(const seed of new Set(finals.map(t=>t.seed)))for(let identity=0;identity<4;identity++)for(let seat=0;seat<4;seat++)assert.equal(finals.filter(t=>t.seed===seed&&t.seatOrder[seat]===identity).length,2);
  assert.equal(challengeLeague(['look+6'],['bid+3','bid+6','bid+9'],2,789,'model').length,8);
});

test('new variants terminate legally and seat permutations preserve identity accounting',()=>{
  const match=playMatch({lineup:['bid+60','bid-liquid+12','look+6','relative+9'],seed:212,seatOrder:[2,0,3,1],validate:true,searchOptions:{samples:2,width:3}});
  assert.deepEqual(match.results.map(r=>r.identity),[2,0,3,1]);assert.equal(match.results.reduce((v,r)=>v+r.win,0),1);
  for(const h of match.auctionHistory){assert.equal(h.winnerSharesBefore.length,4);assert.equal(h.marketBefore.length,4);}
  assert.throws(()=>playMatch({lineup:['bid+3','bid+6','bid+9'],seed:1,seatOrder:[0,0,1]}),/排列/);
  const summary=aggregateMatches([match],'league');assert.ok(summary.every(r=>r.mean95===null));
});

test('relative planner hides opponents private cards and leaves live state unchanged',()=>{
  const a=createGame();for(let i=0;i<4;i++)apply(a,{type:'pass'});apply(a,{type:'setup',buy:-1,goods:[0,1,2],starts:[3,3,3]});
  const b=clone(a);[b.players[1].shares,b.players[2].shares]=[b.players[2].shares,b.players[1].shares];const before=clone(a);
  assert.deepEqual(observation(a,0),observation(b,0));assert.deepEqual(chooseStrategy(a,'relative+9',rng(4),{samples:2,width:3}),chooseStrategy(b,'relative+9',rng(4),{samples:2,width:3}));assert.deepEqual(a,before);
});

test('invasion summaries isolate the single invader and challenger controls',()=>{
  const tasks=pairLeague(['bid+3','bid+6'],2,123,'pair');
  const matches=tasks.map(t=>({...playMatch({...t,validate:true}),...t}));
  const pair=pairDetails(matches)[0];assert.equal(pair.invasions[0].games,8);assert.equal(pair.invasions[1].games,8);assert.equal(pair.invasions[0].win.seedBlocks,2);
  const challenger=challengeDetails(matches.map(m=>({...m,challenger:m.lineup[0]})));assert.equal(challenger.reduce((v,r)=>v+r.n,0),matches.length);
});
