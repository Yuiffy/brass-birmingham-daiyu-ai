const test=require('node:test'),assert=require('node:assert/strict');
process.env.BRASS_RULES='economy-v2';
const I=require('../world_model/human_intent'),S=require('../world_model/simulator'),P=require('../world_model/planner'),E=require('../world_model/encoding');
const model=require('../world_model/experiments/public-human-20261010/human-intent.json');
const fixture=require('../world_model/experiments/public-human-20261010/fixture.json');
test('portable intent MLP agrees with frozen Python softmax fixture',()=>{
 const result=I.predictFeatures(model,fixture.features);
 result.forEach((p,i)=>assert.ok(Math.abs(p-fixture.probability[i])<2e-6));
 assert.ok(Math.abs(result.reduce((a,b)=>a+b,0)-1)<1e-10);
});
test('intent inputs ignore opponents cards, their wild cards, future deck order and RNG',()=>{
 const s=S.create(4,731003),t=S.clone(s),before=I.features(I.fromState(s),0);
 t.players[1].hand=[{type:'wildIndustry'}];t.players[1].hasWildIndustry=true;t.players[1].hasWildLocation=true;
 t.drawDeck.reverse();t._rngState=919;
 assert.deepEqual(I.features(I.fromState(t),0),before);
 assert.deepEqual(I.probabilities(model,t,0),I.probabilities(model,s,0));
});
test('malformed models and normalization fail before inference',()=>{
 for(const mutate of [m=>m.inputScale[0]=0,m=>m.layers[0].weights.pop(),m=>m.layers[0].activation='typo',m=>m.classFrequency[0]=NaN,m=>m.featureNames.reverse()]){
  const bad=structuredClone(model);mutate(bad);assert.throws(()=>I.validate(bad));
 }
});
test('root prior stays bounded, uses legal choices, and zero weight reproduces incumbent',()=>{
 const s=S.create(4,731004),before=S.snapshot(s),world={data:{},valueLayers:[true],estimateValue:(v,p)=>E.value(v,p)};
 const options={type:'guided',world,strategy:'human-card-v2',depth:1,width:4};
 assert.equal(S.key(P.plan(s,options).selected),S.key(P.plan(s,{...options,intentPrior:model,intentWeight:0}).selected));
 const plan=P.plan(s,{...options,intentPrior:model,intentWeight:2});
 assert.ok(plan.branches.every(b=>Math.abs(b.intentBonus)<=6));assert.doesNotThrow(()=>S.step(s,plan.selected));
 assert.deepEqual(S.snapshot(s),before);
 assert.throws(()=>P.plan(s,{...options,intentPrior:model,intentWeight:3}),/bounded/);
});
test('replay audit rejects false provenance, wrong sequence, illegal and incomplete games',async()=>{
 const {auditRecord}=await import('../world_model/audit_public_replays.mjs');
 const state={phase:'playing',turnOrder:[0,1],currentPlayerIdx:0};
 const engine={newGame:()=>state,applyAction:()=>{throw Error('Illegal action');}};
 const record={playerCount:2,seed:1,seats:[{seat:0,isAI:true},{seat:1,isAI:true}],actions:[{seq:0,player:0,action:{type:'pass'}}]};
 const metadata={source:'fixture',split:'diagnostic',sha256:'fixture'};
 assert.throws(()=>auditRecord({...record,seats:[{seat:0,isAI:false,is_ai:1},{seat:1,isAI:true}]},engine,metadata),/provenance/);
 assert.throws(()=>auditRecord({...record,seats:[{seat:4,isAI:true},{seat:1,isAI:true}]},engine,metadata),/provenance/);
 assert.throws(()=>auditRecord({...record,actions:[{seq:1,player:0,action:{type:'pass'}}]},engine,metadata),/order/);
 assert.throws(()=>auditRecord(record,engine,metadata),/Illegal/);
 assert.throws(()=>auditRecord(record,{...engine,applyAction:()=>state},metadata),/Incomplete/);
 const completed=auditRecord(record,{...engine,applyAction:()=>({...state,phase:'game-over',players:[{vp:10},{vp:20}]})},metadata);
 assert.equal(completed.rows.length,0);assert.equal(completed.report.humanActions,0);
});
