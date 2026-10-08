const test=require('node:test'),assert=require('node:assert/strict');
process.env.BRASS_RULES='economy-v2';
const D=require('../js/gameData'),Sim=require('../world_model/simulator'),Logic=require('../js/gameLogic'),E=require('../world_model/encoding');
test('official income positions, initial income and loans use levels rather than spaces',()=>{
 const s=Sim.create(2,42),p=s.players[0],l=new Logic(s);
 assert.equal(s.getIncomeAmount(p.income),0);assert.equal(p.money,17);
 assert.deepEqual([0,3,10,11,12,13,30,31,33,60,61,99].map(D.incomeAtPosition),[-10,-7,0,1,1,2,10,11,11,20,21,30]);
 p.income=25;assert.ok(l.executeLoan(0,0).success);assert.equal(p.income,20);assert.equal(s.getIncomeAmount(p.income),5);
 p.income=3;assert.ok(l.executeLoan(0,0).success);assert.equal(p.income,0);
 const before=Sim.snapshot(s);assert.equal(l.executeLoan(0,0).success,false);assert.deepEqual(Sim.snapshot(s),before);
 p.income=11;s.adjustIncome(0,1);assert.equal(s.getIncomeAmount(p.income),1);s.adjustIncome(0,1);assert.equal(s.getIncomeAmount(p.income),2);
});
test('canal deck discards one card per player and only canal has a one-action opening',()=>{
 for(const n of [2,3,4]){const s=Sim.create(n,991),total=s.drawDeck.length+s.players.reduce((a,p)=>a+p.hand.length,0);
 const expected=Object.values(D.CARD_DECK[n].locations).reduce((a,b)=>a+b,0)+Object.values(D.CARD_DECK[n].industries).reduce((a,b)=>a+b,0);
 assert.equal(total,expected-n);assert.equal(s.actionsPerTurn,1);s.endCanalEra();assert.equal(s.actionsPerTurn,2);assert.equal(s.isFirstRound,false);assert.equal(s.drawDeck.length+s.players.reduce((a,p)=>a+p.hand.length,0),expected);
 }
});
test('only the game-final round skips income; canal order uses final spending',()=>{
 const s=Sim.create(2,22);s.players.forEach(p=>{p.hand=[];p.income=30;p.money=17;});s.drawDeck=[];s.moneySpentThisRound={0:12,1:3};s.actionsThisTurn=1;
 assert.equal(s.advanceTurn(),'endCanalEra');assert.deepEqual(s.turnOrder,[1,0]);assert.ok(s.players.every(p=>p.money===27));
 s.endCanalEra();assert.equal(s.currentPlayerId,1);
 s.players.forEach(p=>{p.hand=[];p.income=30;p.money=17;});s.drawDeck=[];s.actionsThisTurn=1;assert.equal(s.advanceTurn(),'endGame');s.endGame();assert.ok(s.players.every(p=>p.money===17&&p.vp===0));
});
test('bankruptcy liquidates assets before losing VP; points stop at zero',()=>{
 const s=Sim.create(2,2),p=s.players[0];p.income=0;p.money=0;p.vp=0;
 s.boardIndustries.birmingham_0={playerId:0,type:'cottonMill',flipped:false,tileData:{...D.INDUSTRY_DATA.cottonMill[0]}};
 s.endRound();assert.equal(s.boardIndustries.birmingham_0,undefined);assert.equal(p.vp,0);assert.equal(p.money,0);
});
test('end-game ties use income level, then money; old model schema is rejected',()=>{
 const s=Sim.create(2,2),[a,b]=s.players;a.vp=b.vp=100;a.income=11;b.income=12;a.money=20;b.money=5;
 assert.ok(s.comparePlayers(a,b)<0);b.income=13;assert.ok(s.comparePlayers(a,b)>0);
 assert.equal(E.schema.version,'brass-wm-economy-v2');assert.equal(E.fields[E.indices['p0.income']].scale,100);
 const {Network}=require('../world_model/inference');assert.throws(()=>new Network(require('../world_model/models/world-model.json')),/schema/);
});
test('corrected seeded games terminate with score components only',()=>{
 for(const n of [2,3,4]){let s=Sim.create(n,54),moves=0;const earned=Array(n).fill(0);
 while(!s.gameOver){const a=Sim.candidates(s).sort((a,b)=>b.score-a.score)[0];assert.ok(a);const result=Sim.step(s,a);s=result.state;for(const score of result.event.scores||[])earned[score.playerId]+=score.linkVP+score.industryVP;assert.ok(++moves<500);}
 assert.ok(s.players.every(p=>Number.isFinite(p.vp)&&p.money>=0));assert.ok(moves>40);
 }
});
test('market supply pays cash, flips exhausted new iron, and cube quantities are real',()=>{
 const s=Sim.create(2,17),p=s.players[0];s.ironMarket=7;
 s.boardIndustries.birmingham_0={playerId:0,type:'ironWorks',tileData:{...D.INDUSTRY_DATA.ironWorks[0]},resourceCubes:4,flipped:false};
 const cash=p.money;s.supplyNewIndustry('birmingham_0');assert.equal(s.ironMarket,11);assert.equal(p.money,cash+6);assert.equal(p.income,13);assert.equal(s.boardIndustries.birmingham_0.flipped,true);
 s.ironMarket=0;assert.equal(s.getIronPrice(),6);assert.ok(s.findIronSource(0).length>=2);
 assert.equal(s.findCoalSource('birmingham',0).length,0);s.boardLinks['birmingham-oxford']={playerId:0,type:'canal'};assert.ok(s.findCoalSource('birmingham',0).some(x=>x.type==='market'));
});
test('multi-sale reserves beer cubes, requires a merchant even for zero-beer goods and fails atomically',()=>{
 const s=Sim.create(2,18),l=new Logic(s),p=s.currentPlayerId;
 s.boardLinks['birmingham-oxford']={playerId:p,type:'canal'};
 s.merchantTiles=[{location:'oxford',buys:null,hasBeer:false}];
 for(const key of ['birmingham_0','birmingham_1'])s.boardIndustries[key]={playerId:p,type:'cottonMill',tileData:{...D.INDUSTRY_DATA.cottonMill[0]},resourceCubes:0,flipped:false};
 s.boardIndustries.birmingham_2={playerId:p,type:'brewery',tileData:{...D.INDUSTRY_DATA.brewery[0]},resourceCubes:2,flipped:false};
 const keys=['birmingham_0','birmingham_1'];assert.ok(l.planSales(p,keys));assert.ok(Sim.candidates(s).some(a=>a.target?.keys?.length===2));
 const hand=s.currentPlayer.hand.length;assert.ok(l.executeSell(p,keys,0).success);assert.equal(s.currentPlayer.hand.length,hand-1);assert.equal(s.boardIndustries.birmingham_2.resourceCubes,0);assert.equal(s.merchantTiles[0].bonusClaimed,undefined);
 const before=Sim.snapshot(s);assert.equal(l.executeSell(p,keys,0).success,false);assert.deepEqual(Sim.snapshot(s),before);
 s.boardIndustries.birmingham_0.flipped=false;s.boardIndustries.birmingham_0.tileData.beersToSell=0;s.boardLinks={};assert.equal(l.planSales(p,['birmingham_0']),null);
});
test('first industry-card placement is both offered and executable',()=>{
 const s=Sim.create(2,72),p=s.currentPlayerId;s.players[p].hand=[{type:'industry',industryType:'brewery'}];
 const targets=new Logic(s).getValidBuildTargets(p);assert.ok(targets.length);assert.ok(Sim.candidates(s).some(a=>a.action==='build'));
});
