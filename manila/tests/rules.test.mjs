import test from 'node:test';
import assert from 'node:assert/strict';
import { createGame, apply, legalActions, transition, wealth, capacity, rng, clone, observation, determinize, redeem, startsList, GOODS } from '../engine.mjs';
import { policy, diceProbability, countProbabilities, analyzeAuction, ruleWorldModel, simulateGame } from '../ai.mjs';

function prepared(count=4) { const s=createGame(count,1234); for(let i=0;i<count;i++) apply(s,{type:'pass'}); apply(s,{type:'setup',buy:-1,goods:[0,1,2],starts:[3,3,3]}); return s; }
function rollState(roll=2) { const s=prepared(); s.phase='roll';s.rolls=roll;s.boats.forEach(b=>{b.pos=11;b.crew=[0];}); return s; }
const die = n => () => (n-.5)/6;

test('initial deck uses only 3 of each cargo, two private shares, total supply conserved',()=>{
  for(const n of [3,4,5]) for(let seed=1;seed<20;seed++) { const s=createGame(n,seed);assert.deepEqual(s,createGame(n,seed));for(const p of s.players){assert.equal(p.cash,30);assert.equal(p.shares.reduce((a,b)=>a+b),2);}for(let g=0;g<4;g++){const dealt=s.players.reduce((v,p)=>v+p.shares[g],0);assert.ok(dealt<=3);assert.equal(s.supply[g]+dealt,5);} }
});
test('all-pass retains previous captain for free; passed player cannot reenter',()=>{
  const s=createGame();s.captain=2;s.actor=2;for(let i=0;i<4;i++)apply(s,{type:'pass'});assert.equal(s.captain,2);assert.equal(s.players[2].cash,30);assert.equal(s.phase,'setup');
  const t=createGame();apply(t,{type:'pass'});apply(t,{type:'bid',amount:1});apply(t,{type:'bid',amount:2});apply(t,{type:'pass'});assert.equal(t.actor,1);assert.throws(()=>apply(t,{type:'bid',amount:2}));
});
test('winning payment forces mortgages; credit twelve and terminal debt fifteen',()=>{
  const s=createGame();const before=wealth(s,0);apply(s,{type:'bid',amount:40});for(let i=0;i<3;i++)apply(s,{type:'pass'});assert.equal(s.players[0].cash,2);assert.equal(s.players[0].mortgages.reduce((a,b)=>a+b),1);assert.equal(wealth(s,0),before-43);assert.equal(capacity(s.players[0]),14);
});
test('bids exceeding capacity and zero bids are rejected without mutation',()=>{
  const s=createGame(),before=clone(s);assert.throws(()=>apply(s,{type:'bid',amount:55}));assert.throws(()=>apply(s,{type:'bid',amount:0}));assert.deepEqual(s,before);
});
test('setup bounds, total nine, unique cargo and single stock purchase',()=>{
  const s=createGame();for(let i=0;i<4;i++)apply(s,{type:'pass'});const before=clone(s);
  for(const a of [{type:'setup',buy:-1,goods:[0,1,2],starts:[3,3,4]},{type:'setup',buy:-1,goods:[0,0,2],starts:[3,3,3]},{type:'setup',buy:-1,goods:[0,1,2],starts:[6,3,0]}])assert.throws(()=>apply(s,a));
  assert.deepEqual(s,before);apply(s,{type:'setup',buy:0,goods:[0,1,2],starts:[3,3,3]});assert.equal(s.players[0].cash,25);assert.equal(s.players[0].shares[0],before.players[0].shares[0]+1);assert.equal(s.supply[0],before.supply[0]-1);assert.ok(startsList().every(a=>a.reduce((v,n)=>v+n,0)===9&&a.every(n=>n<=5&&n>=0)));
});
test('three players deploy twice before first roll and four times total',()=>{
  const s=prepared(3);for(let i=0;i<3;i++)apply(s,{type:'stop'});assert.equal(s.phase,'placement');for(let i=0;i<3;i++)apply(s,{type:'skip'});assert.equal(s.phase,'roll');apply(s,{type:'roll'},die(1));assert.equal(s.rolls,1);for(let i=0;i<3;i++)apply(s,{type:'skip'});apply(s,{type:'roll'},die(1));for(let i=0;i<3;i++)apply(s,{type:'skip'});assert.equal(s.wave,4);assert.equal(s.phase,'roll');
});
test('stopping placement is permanent within voyage; boat uses cheapest free spot',()=>{
  const s=prepared();assert.equal(legalActions(s).find(a=>a.where==='boat'&&a.index===0).cost,2);apply(s,legalActions(s).find(a=>a.where==='boat'&&a.index===0));assert.equal(legalActions(s).find(a=>a.where==='boat'&&a.index===0).cost,3);apply(s,{type:'stop'});apply(s,{type:'stop'});apply(s,{type:'stop'});apply(s,{type:'roll'},die(1));apply(s,{type:'stop'});assert.deepEqual(legalActions(s),[{type:'skip'}]);
});
test('third-roll thirteen docks without pirates, but is plundered with pirates',()=>{
  const s=rollState();apply(s,{type:'roll'},die(2));assert.ok(s.boats.every(b=>b.destination==='port'));assert.equal(s.phase,'settlement');
  const t=rollState();t.pirates=[1,null];apply(t,{type:'roll'},die(2));assert.equal(t.phase,'pirate-route');assert.equal(t.actor,1);for(let i=0;i<3;i++)apply(t,{type:'route',destination:'yard'});assert.ok(t.boats.every(b=>b.plundered&&b.destination==='yard'));const before=t.players[1].cash;apply(t,{type:'settle'});assert.equal(t.players[1].cash,before+72);assert.deepEqual(t.market,[0,0,0,0]);
});
test('second-roll pirates board empty slots, promotion and subsequent rehire are supported',()=>{
  const s=rollState(1);s.pirates=[1,2];apply(s,{type:'roll'},die(2));assert.equal(s.phase,'boarding');apply(s,{type:'board',index:0});assert.equal(s.actor,2);apply(s,{type:'board',index:-1});assert.deepEqual(s.pirates,[2,null]);assert.deepEqual(s.boats[0].crew,[0,1]);assert.equal(s.phase,'placement');assert.ok(legalActions(s).some(a=>a.where==='pirates'&&a.index===1));
});
test('no boarding of full or docked boats; arrived boat unavailable for new crew',()=>{
  const s=rollState(1);s.pirates=[1,null];s.boats[0].crew=[0,0,0];s.boats[1].pos=14;s.boats[1].destination='port';apply(s,{type:'roll'},die(2));assert.ok(!legalActions(s).some(a=>a.type==='board'&&(a.index===0||a.index===1)));apply(s,{type:'board',index:-1});assert.ok(!legalActions(s).some(a=>a.where==='boat'&&a.index===1));
});
test('small pilot moves before large, crossing thirteen docks immediately, no pirate trigger',()=>{
  const s=prepared();s.wave=2;s.rolls=2;s.turnInWave=3;s.actor=3;s.pilots=[1,2];s.boats[0].pos=13;s.boats[1].pos=12;s.pirates=[0,null];apply(s,{type:'stop'});assert.equal(s.phase,'pilot-small');assert.equal(s.actor,1);apply(s,{type:'pilot',moves:[[0,1]]});assert.equal(s.boats[0].destination,'port');assert.equal(s.phase,'pilot-large');assert.equal(s.actor,2);assert.ok(!legalActions(s).some(a=>a.moves.some(([i])=>i===0)));apply(s,{type:'pilot',moves:[[1,1]]});assert.equal(s.boats[1].pos,13);assert.equal(s.boats[1].destination,null);assert.equal(s.phase,'roll');
});
test('insurance pays 6+8+15 for empty shipyard positions and receives profits first',()=>{
  const s=rollState();s.phase='settlement';s.insurer=0;s.players[0].cash=40;s.boats.forEach(b=>b.destination='yard');apply(s,{type:'settle'});assert.equal(s.players[0].cash,11);
  const t=rollState();t.phase='settlement';t.insurer=0;t.players[0].cash=0;t.boats[0].destination='port';t.boats[1].destination='yard';t.boats[2].destination='yard';apply(t,{type:'settle'});assert.equal(t.players[0].cash,10);assert.equal(t.players[0].mortgages.reduce((a,b)=>a+b),0);
});
test('insurance does not pay itself; insolvency is covered by bank after forced loans',()=>{
  const s=rollState();s.phase='settlement';s.insurer=0;s.players[0].cash=10;s.yard[0]=0;s.yard[1]=1;s.boats.forEach(b=>b.destination='yard');s.players[0].mortgages=[...s.players[0].shares];const before=s.players[1].cash;apply(s,{type:'settle'});assert.equal(s.players[0].cash,0);assert.equal(s.players[1].cash,before+8);
});
test('blind passenger costs remaining cash, excludes insurance and requires exhausted credit',()=>{
  const s=prepared();s.players[0].cash=0;s.players[0].mortgages=[...s.players[0].shares];const legal=legalActions(s);assert.ok(!legal.some(a=>a.where==='insurance'));assert.ok(legal.filter(a=>a.type==='place').every(a=>a.cost===0&&a.blind));apply(s,legal.find(a=>a.where==='boat'));assert.equal(s.players[0].cash,0);
});
test('stock rises 20 to 30 and ends game; plunder routing to port rises stock',()=>{
  const s=rollState();s.market[0]=4;s.pirates=[1,null];apply(s,{type:'roll'},die(2));for(let i=0;i<3;i++)apply(s,{type:'route',destination:'port'});apply(s,{type:'settle'});assert.equal(s.market[0],5);assert.equal(s.phase,'finished');assert.equal(s.players[1].cash,102);assert.deepEqual(legalActions(s),[]);
});
test('redeeming needs fifteen cash and preserves total wealth',()=>{
  const s=createGame(),g=s.players[0].shares.findIndex(n=>n);s.players[0].mortgages[g]=1;const before=wealth(s,0);redeem(s,0,g);assert.equal(s.players[0].cash,15);assert.equal(wealth(s,0),before);assert.throws(()=>redeem(s,0,g));
});
test('private observations redact holdings, seed and derived historical wealth',()=>{
  const s=createGame();s.history=[{voyage:1,wealth:[1,2,3,4],deltaAtSettlement:[1,1,1,1]}];const o=observation(s,0);assert.deepEqual(o.players[0].shares,s.players[0].shares);assert.equal(o.players[1].shares,null);assert.equal(o.seed,undefined);assert.equal(o.history[0].wealth,undefined);assert.equal(o.history[0].deltaAtSettlement,undefined);
});
test('determinization preserves own information, public purchases, counts and supply',()=>{
  const s=createGame();const a=determinize(s,0,rng(1)),b=determinize(s,0,rng(2));assert.deepEqual(a.players[0],s.players[0]);assert.notDeepEqual(a.players.slice(1),b.players.slice(1));for(let g=0;g<4;g++)assert.equal(a.supply[g]+a.players.reduce((v,p)=>v+p.shares[g],0),5);
});
test('dice DP matches exhaustive 216 outcomes; port-count probability sums to one',()=>{
  for(let start=0;start<=5;start++){let hits=0;for(let a=1;a<=6;a++)for(let b=1;b<=6;b++)for(let c=1;c<=6;c++)if(start+a+b+c>=13)hits++;assert.ok(Math.abs(diceProbability(start,3)-hits/216)<1e-12);}
  assert.ok(Math.abs(countProbabilities([.2,.5,.9]).reduce((a,b)=>a+b)-1)<1e-12);
});
test('rule-world-model step is pure and rewards match engine wealth differences',()=>{
  const s=createGame(),before=clone(s),a={type:'bid',amount:1},r=ruleWorldModel.step(s,a,rng(1));assert.deepEqual(s,before);assert.equal(r.next.bid,1);assert.deepEqual(r.rewards,[0,0,0,0]);assert.equal(ruleWorldModel.observe(s,0).players[1].shares,null);
});
test('full games terminate across 3/4/5 players with no cash or share violations',()=>{
  for(const n of [3,4,5])for(let seed=1;seed<=6;seed++){
    const s=createGame(n,seed),random=rng(seed);let steps=0;
    while(s.phase!=='finished'&&steps++<4000){apply(s,policy(s,'balanced',random),random);for(const p of s.players){assert.ok(p.cash>=0);assert.ok(p.mortgages.every((m,g)=>m<=p.shares[g]));}for(let g=0;g<4;g++)assert.equal(s.supply[g]+s.players.reduce((v,p)=>v+p.shares[g],0),5);}
    assert.equal(s.phase,'finished');assert.ok(s.market.some(n=>n===5));assert.ok(s.history.length>=5);
  }
});
test('counterfactual analysis reproducible, paired CI ordered, source untouched and low bids nested',async()=>{
  const s=createGame(),before=clone(s),options={samples:16,maxBid:3,seed:77};const a=await analyzeAuction(s,0,options),b=await analyzeAuction(s,0,options);assert.deepEqual(a,b);assert.deepEqual(s,before);assert.equal(a.rows.length,3);assert.ok(a.rows.every(r=>r.lower<=r.delta&&r.delta<=r.upper&&r.winRate===null));assert.ok(Math.abs(a.rows[0].delta-a.rows[1].delta-1)<1e-9);await assert.rejects(()=>analyzeAuction(s,1));
});
test('terminal evaluator only reports win rates for actual terminal games',async()=>{
  const r=await analyzeAuction(createGame(),0,{samples:16,maxBid:1,horizon:0});assert.ok(r.rows[0].winRate>=0&&r.rows[0].winRate<=1);assert.ok(r.baselineWin>=0&&r.baselineWin<=1);
});
test('hidden-hand swaps cannot change advice or analysis under identical public information',async()=>{
  const a=createGame(),b=clone(a);[b.players[1].shares,b.players[2].shares]=[b.players[2].shares,b.players[1].shares];assert.deepEqual(observation(a,0),observation(b,0));assert.deepEqual(await analyzeAuction(a,0,{samples:16,maxBid:1}),await analyzeAuction(b,0,{samples:16,maxBid:1}));
});
