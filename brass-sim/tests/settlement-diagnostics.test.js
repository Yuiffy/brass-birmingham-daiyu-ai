const test=require('node:test'),assert=require('node:assert/strict');
process.env.BRASS_RULES='economy-v2';
const S=require('../world_model/simulator'),D=require('../world_model/settlement_diagnostics'),Data=require('../js/gameData');
test('settlement snapshots precede canal level-one removal and distinguish unsold resources',()=>{
 let s=S.create(4,7132);s.players.forEach(p=>p.hand=[]);s.currentPlayer.hand=[{type:'location',location:'birmingham'}];s.drawDeck=[];
 s.boardIndustries.birmingham_0={type:'cottonMill',playerId:0,flipped:false,resourceCubes:0,tileData:Data.INDUSTRY_DATA.cottonMill[0]};
 s.boardIndustries.birmingham_1={type:'coalMine',playerId:0,flipped:false,resourceCubes:3,tileData:Data.INDUSTRY_DATA.coalMine[0]};
 const next=S.step(s,S.candidates(s).find(a=>a.action==='pass'),{captureSettlement:true});
 assert.equal(next.state.era,'rail');assert.equal(next.state.boardIndustries.birmingham_0,undefined);
 const before=Object.assign(S.clone(next.state),next.event.beforeSettlement),r=D.settlement(before)[0];
 assert.equal(r.unsoldIndustries,1);assert.equal(r.unflippedResources,1);assert.ok(r.unsoldVP>0);
});
test('development conversion requires a later higher-level build of that industry',()=>{
 const trace=[{seq:0,rawAction:{action:'develop'},developed:[{type:'pottery',level:1},{type:'cottonMill',level:1}]},
  {seq:1,rawAction:{action:'build',target:{industryType:'pottery'}},builtLevel:1},
  {seq:2,rawAction:{action:'build',target:{industryType:'cottonMill'}},builtLevel:2}];
 assert.deepEqual(D.unconvertedDevelopment(trace).map(x=>x.type),['pottery']);
});
