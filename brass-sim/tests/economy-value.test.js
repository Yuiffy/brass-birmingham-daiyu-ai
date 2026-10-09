const test=require('node:test'),assert=require('node:assert/strict');
process.env.BRASS_RULES='economy-v2';
const Sim=require('../world_model/simulator'),E=require('../world_model/encoding'),V=require('../world_model/value_features');
const P=require('../world_model/planner'),{tournament}=require('../world_model/tournament');
test('paired tournaments route guided agents to the candidate and world agents to the frozen model',async()=>{
 const original=P.plan,calls=[],world={data:{kind:'frozen-world'}},guidedWorld={valueLayers:true,data:{kind:'candidate-value'}};
 P.plan=(state,config)=>{calls.push({type:config.type,world:config.world});return {selected:Sim.candidates(state).find(a=>a.action==='pass'),elapsedMs:0};};
 try{const r=await tournament({games:2,types:['guided','world'],seed:78,world,guidedWorld,depth:1,width:8});
  assert.ok(calls.some(c=>c.type==='world'));assert.ok(calls.some(c=>c.type==='guided'));
  assert.ok(calls.every(c=>c.world===(c.type==='guided'?guidedWorld:world)));
  assert.equal(r.config.rulesVersion,'economy-v2');assert.equal(r.completedGames,2);
  assert.ok(r.games.every(g=>g.scores.every(s=>s.incomeBonus===0&&s.vpWithoutIncomeBonus===s.vp)));
  assert.equal(r.rows.reduce((sum,row)=>sum+row.wins,0),2);
 }finally{P.plan=original;}
});
test('score anchor equals observable current-era settlement and does not double-count terminal scores',()=>{
 const D=require('../js/gameData'),s=Sim.create(3,78);s.players[0].vp=12;
 s.boardIndustries.birmingham_0={playerId:0,type:'cottonMill',tileData:D.INDUSTRY_DATA.cottonMill[1],flipped:true,resourceCubes:0};
 s.boardLinks['birmingham-oxford']={playerId:0,type:'canal'};
 const expected=Sim.clone(s).calculateEraScore()[0].totalVP;
 assert.ok(Math.abs(V.scoreAnchor(E.encodeState(s),0)-expected)<1e-6);
 s.gameOver=true;assert.equal(V.scoreAnchor(E.encodeState(s),0),12);
});
