const test=require('node:test'),assert=require('node:assert/strict');
process.env.BRASS_RULES='economy-v2';
const D=require('../js/gameData'),Sim=require('../world_model/simulator'),Logic=require('../js/gameLogic'),E=require('../world_model/encoding');
function fixture(){
 const s=Sim.create(4,203010001);s.endCanalEra();s.currentPlayerIndex=s.turnOrder.indexOf(0);
 s.players[0].money=50;s.players[0].hand=[{type:'location',location:'birmingham'},{type:'location',location:'coventry'}];
 s.boardIndustries.birmingham_0={playerId:0,type:'brewery',tileData:{...D.INDUSTRY_DATA.brewery[1]},flipped:false,resourceCubes:2};
 return s;
}
test('new single rail can reach market coal across the link it just placed',()=>{
 const s=fixture(),l=new Logic(s),before=Sim.snapshot(s),plan=l.planNetwork(0,['birmingham-oxford']);
 assert.ok(plan);assert.equal(plan.cost,5+s.getCoalPrice());assert.deepEqual(Sim.snapshot(s),before);
 assert.ok(l.executeNetwork(0,'birmingham-oxford',0).success);assert.equal(s.players[0].hand.length,1);
 assert.equal(s.coalMarket,before.coalMarket-1);
});
test('double rail uses one action/card, £15 plus sequential coal prices and one brewery beer',()=>{
 const s=fixture(),l=new Logic(s),ids=['birmingham-oxford','birmingham-coventry'],before=Sim.snapshot(s);
 const plan=l.planNetwork(0,ids);assert.ok(plan);assert.equal(plan.cost,15+D.COAL_MARKET_PRICES[D.COAL_MARKET_PRICES.length-s.coalMarket]+D.COAL_MARKET_PRICES[D.COAL_MARKET_PRICES.length-s.coalMarket+1]);
 assert.ok(l.executeNetworkLinks(0,ids,0).success);
 assert.equal(s.players[0].money,50-plan.cost);assert.equal(s.players[0].hand.length,1);
 assert.equal(s.players[0].linksRemaining.rail,before.players[0].linksRemaining.rail-2);
 assert.equal(s.boardIndustries.birmingham_0.resourceCubes,1);assert.equal(s.coalMarket,before.coalMarket-2);
});
test('double rails cannot use a later link to supply the first link retroactively',()=>{
 const s=fixture(),l=new Logic(s),before=Sim.snapshot(s);
 assert.equal(l.planNetwork(0,['birmingham-coventry','birmingham-oxford']),null);
 assert.equal(l.executeNetworkLinks(0,['birmingham-coventry','birmingham-oxford'],0).success,false);
 assert.deepEqual(Sim.snapshot(s),before);
});
test('a second rail can extend the first; opponent beer must reach the second link',()=>{
 const s=fixture(),l=new Logic(s);delete s.boardIndustries.birmingham_0;
 s.boardLinks['birmingham-oxford']={playerId:0,type:'rail'};
 s.boardIndustries.nuneaton_0={playerId:1,type:'brewery',tileData:{...D.INDUSTRY_DATA.brewery[1]},flipped:false,resourceCubes:2};
 assert.ok(l.planNetwork(0,['birmingham-coventry','coventry-nuneaton']));
 assert.equal(l.planNetwork(0,['birmingham-dudley','birmingham-walsall']),null);
});
test('merchant beer, duplicate links, absent cards and shortages fail without mutation',()=>{
 const s=fixture(),l=new Logic(s),ids=['birmingham-oxford','birmingham-coventry'];
 s.boardIndustries.birmingham_0.flipped=true;s.merchantTiles=[{location:'oxford',buys:null,hasBeer:true}];
 let before=Sim.snapshot(s);assert.equal(l.executeNetworkLinks(0,ids,0).success,false);assert.deepEqual(Sim.snapshot(s),before);
 s.boardIndustries.birmingham_0.flipped=false;
 for(const [links,card] of [[['birmingham-oxford','birmingham-oxford'],0],[ids,99]]){
  before=Sim.snapshot(s);assert.equal(l.executeNetworkLinks(0,links,card).success,false);assert.deepEqual(Sim.snapshot(s),before);
 }
 s.players[0].money=15;before=Sim.snapshot(s);assert.equal(l.executeNetworkLinks(0,ids,0).success,false);assert.deepEqual(Sim.snapshot(s),before);
 s.players[0].money=50;s.players[0].linksRemaining.rail=1;assert.equal(l.planNetwork(0,ids),null);
 s.era='canal';assert.equal(l.planNetwork(0,ids),null);
});
test('candidate doubles validate and encode both links without modifying legacy candidate selection',()=>{
 const s=fixture(),before=Sim.snapshot(s),normal=Sim.candidates(s),expanded=Sim.candidates(s,{doubleRail:true});
 assert.ok(normal.every(a=>!a.target?.connectionIds));
 const a=expanded.find(a=>a.target?.connectionIds?.join()===['birmingham-oxford','birmingham-coventry'].join());assert.ok(a);
 const encoded=E.encodeAction(a,s);assert.equal(encoded.length,E.encodeAction(normal[0],s).length);
 assert.ok(Sim.label(a).includes(' + '));const result=Sim.step(s,a);assert.deepEqual(Sim.snapshot(s),before);
 assert.equal(result.state.actionsThisTurn,1);assert.ok(a.target.connectionIds.every(id=>result.state.boardLinks[id]));
});

test('successful planning isolates spending, resource flips and income from the original',()=>{
 const s=fixture();s.boardIndustries.birmingham_0.resourceCubes=1;
 const before=Sim.snapshot(s),plan=new Logic(s).planNetwork(0,['birmingham-oxford','birmingham-coventry']);
 assert.ok(plan.state.boardIndustries.birmingham_0.flipped);
 assert.ok(plan.state.players[0].income>s.players[0].income);
 assert.deepEqual(Sim.snapshot(s),before);
});

test('new model action encoding distinguishes rail order and brewery source without changing legacy',()=>{
 const s=fixture();const ids=['birmingham-oxford','birmingham-coventry'];
 const a={action:'network',cardIndex:0,target:{connectionIds:ids,beerKey:'birmingham_0'}};
 const b={...a,target:{...a.target,connectionIds:ids.slice().reverse()}};
 assert.deepEqual(E.encodeAction(a,s),E.encodeAction(b,s));
 const options={version:'resource-network-v2'};
 assert.notDeepEqual(E.encodeAction(a,s,options),E.encodeAction(b,s,options));
 const c={...a,target:{...a.target,beerKey:'nuneaton_0'}};
 assert.notDeepEqual(E.encodeAction(a,s,options),E.encodeAction(c,s,options));
 const single=Sim.candidates(s)[0];assert.deepEqual(E.encodeAction(single,s,options),E.encodeAction(single,s));
 assert.equal(E.encodeAction(a,s,options).length,E.encodeAction(a,s).length);
 assert.ok(E.schema.links.length<64);assert.equal(E.schema.types.length,6);
});

test('canal overbuild replaces an owned lower tier while preserving one tile per city',()=>{
 const s=Sim.create(4,203010020),p=s.players[0];p.money=100;
 s.currentPlayerIndex=s.turnOrder.indexOf(0);
 p.hand=[{type:'location',location:'birmingham'}];
 s.boardLinks['birmingham-oxford']={playerId:0,type:'canal'};
 const stack=p.industryTiles.ironWorks;stack[0].used=true;
 s.boardIndustries.birmingham_2={playerId:0,type:'ironWorks',tileData:{...D.INDUSTRY_DATA.ironWorks[0]},flipped:true,resourceCubes:0};
 const l=new Logic(s),targets=l.getValidBuildTargets(0).filter(t=>t.cityId==='birmingham');
 assert.ok(targets.some(t=>t.slotIndex===2&&t.industryType==='ironWorks'));
 assert.ok(targets.every(t=>t.slotIndex===2));
 const a=Sim.candidates(s).find(a=>a.action==='build'&&a.target.cityId==='birmingham'&&a.target.industryType==='ironWorks');
 assert.ok(a);const next=Sim.step(s,a).state;
 assert.equal(next.boardIndustries.birmingham_2.tileData.level,2);
 assert.equal(Object.values(next.boardIndustries).filter(t=>t.playerId===0).length,1);
});

test('depleted opponent resources require the same industry and a strictly higher tier',()=>{
 const s=Sim.create(4,203010021),p=s.players[0];p.money=100;p.hand=[{type:'location',location:'birmingham'}];
 s.boardLinks['birmingham-oxford']={playerId:0,type:'canal'};s.ironMarket=0;
 s.boardIndustries.birmingham_2={playerId:1,type:'ironWorks',tileData:{...D.INDUSTRY_DATA.ironWorks[0]},flipped:true,resourceCubes:0};
 const target=()=>new Logic(s).getValidBuildTargets(0).find(t=>t.cityId==='birmingham'&&t.slotIndex===2);
 assert.equal(target(),undefined);p.industryTiles.ironWorks[0].used=true;assert.ok(target());
 s.boardIndustries.birmingham_2.type='coalMine';s.coalMarket=0;assert.equal(target(),undefined);
});
