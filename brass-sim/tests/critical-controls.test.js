const test=require('node:test'),assert=require('node:assert/strict');
process.env.BRASS_RULES='economy-v2';
const {Network}=require('../world_model/inference'),P=require('../world_model/planner'),S=require('../world_model/simulator');
const artifact=require('../world_model/experiments/public-human-20261010/critical-world-model.json');
const fixture=require('../world_model/experiments/public-human-20261010/critical-fixture.json');
test('trained expanded categorical controls match Python prediction',()=>{
 const model=new Network(artifact),result=model.predict(fixture.state,fixture.action);
 assert.ok(Math.max(...result.map((x,i)=>Math.abs(x-fixture.prediction[i])))<2e-5);
 for(const h of artifact.controlSpec.heads)for(const i of h.indices)assert.equal(result[i],fixture.prediction[i]);
});
test('new world checkpoint plans recursively without real-engine successor repairs',()=>{
 const state=S.create(4,74219),snapshot=S.snapshot(state),original=S.step;
 S.step=()=>{throw Error('Real successor forbidden');};
 try{
  const plan=P.plan(state,{type:'world',world:new Network(artifact),depth:2,width:4,doubleRail:true});
  assert.ok(plan.selected);assert.deepEqual(S.snapshot(state),snapshot);
 }finally{S.step=original;}
});
