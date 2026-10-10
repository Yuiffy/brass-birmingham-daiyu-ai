const test=require('node:test'),assert=require('node:assert/strict');
process.env.BRASS_RULES='economy-v2';
const D=require('../js/gameData'),S=require('../world_model/simulator'),H=require('../world_model/human_strategy'),P=require('../world_model/planner'),E=require('../world_model/encoding');
function cotton(state,key='birmingham_0',level=2){
 state.boardIndustries[key]={playerId:0,type:'cottonMill',tileData:{...D.INDUSTRY_DATA.cottonMill.find(t=>t.level===level)},resourceCubes:0,flipped:true};
}
test('canal strategy values persistent industries across both eras and ignores terminal cash',()=>{
 const s=S.create(4,203010010);cotton(s);const canal=H.components(s,0).industry;
 s.era='rail';const rail=H.components(s,0).industry;assert.ok(canal>rail);
 s.gameOver=true;s.players[0].vp=140;assert.equal(H.components(s,0).total,140);
 s.players[0].money=1000;s.players[0].income=99;assert.equal(H.components(s,0).total,140);
});
test('links earn printed merchant and actual adjacent industry icons, not a fixed link-count bonus',()=>{
 const s=S.create(4,203010011);cotton(s);s.boardLinks['birmingham-oxford']={playerId:0,type:'canal'};
 assert.equal(H.components(s,0).links,4);s.boardIndustries.birmingham_0.tileData.linkVP=1;
 assert.equal(H.components(s,0).links,3);
});
test('sale readiness depends on merchant reachability and affordable remaining actions',()=>{
 const s=S.create(4,203010012);cotton(s);s.boardIndustries.birmingham_0.flipped=false;
 s.boardIndustries.birmingham_0.tileData.beersToSell=0;s.merchantTiles=[{location:'oxford',buys:'cottonMill',hasBeer:false}];
 const before=H.components(s,0).industry;s.boardLinks['birmingham-oxford']={playerId:0,type:'canal'};
 assert.ok(H.components(s,0).industry>before);
 s.players[0].hand=[];s.drawDeck=[];assert.equal(H.components(s,0).industry,0);
});
test('resource surplus lowers estimated coal conversion rather than rewarding idle cubes',()=>{
 const s=S.create(4,203010013);s.era='rail';s.boardLinks['birmingham-oxford']={playerId:0,type:'rail'};
 s.boardIndustries.birmingham_2={playerId:0,type:'coalMine',tileData:{...D.INDUSTRY_DATA.coalMine[2]},resourceCubes:4,flipped:false};
 const before=H.components(s,0).industry;
 for(let i=0;i<4;i++)s.boardIndustries['coventry_'+i]={playerId:1,type:'coalMine',tileData:{...D.INDUSTRY_DATA.coalMine[3]},resourceCubes:5,flipped:false};
 assert.ok(H.components(s,0).industry<before);
});
test('guided strategy uses learned estimates, cannot read future deck order, and leaves state intact',()=>{
 const s=S.create(2,203010014),before=S.snapshot(s);let calls=0;
 const world={data:{planningCandidates:'learned-pool-v1'},valueLayers:[true],
  estimateValue:(v,p)=>{calls++;return E.value(v,p);}};
 const run=state=>P.plan(state,{type:'guided',world,strategy:'human-guide-v1',depth:1,width:4});
 const a=run(s);assert.ok(calls>0);assert.ok(a.selected);assert.deepEqual(S.snapshot(s),before);
 const t=S.clone(s);t.drawDeck.reverse();t._rngState=777;
 assert.equal(S.key(run(t).selected),S.key(a.selected));assert.doesNotThrow(()=>S.step(s,a.selected));
 assert.throws(()=>P.plan(s,{type:'world',world,strategy:'human-guide-v1'}),/guided/);
});
