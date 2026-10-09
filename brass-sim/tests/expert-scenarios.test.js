// Synthetic positions inspired by published WBC 2023 decisions, not reconstructed
// human replays. Source: https://boardgamers.org/yearbook23/brs.html
const test=require('node:test'),assert=require('node:assert/strict');
process.env.BRASS_RULES='economy-v2';
const S=require('../world_model/simulator'),Logic=require('../js/gameLogic'),D=require('../js/gameData');
function beerRace(){
    const s=S.create(4,312010022);
    s.boardLinks['birmingham-oxford']={playerId:0,type:'canal'};
    s.boardLinks['birmingham-coventry']={playerId:0,type:'canal'};
    s.boardLinks['coventry-nuneaton']={playerId:0,type:'canal'};
    s.boardIndustries.birmingham_0={playerId:0,type:'cottonMill',tileData:{...D.INDUSTRY_DATA.cottonMill[0]},resourceCubes:0,flipped:false};
    s.boardIndustries.nuneaton_0={playerId:1,type:'brewery',tileData:{...D.INDUSTRY_DATA.brewery.find(t=>t.level===2)},resourceCubes:1,flipped:false};
    s.merchantTiles=[{location:'oxford',buys:'cottonMill',hasBeer:false}];
    s.players[0].hand=[{type:'location',location:'birmingham'}];
    s.players[1].money=90;
    return s;
}
test('WBC-inspired canal sale consumes persistent rival beer and denies its rail double',()=>{
    const before=beerRace(),after=S.clone(before);
    assert.ok(new Logic(after).executeSell(0,['birmingham_0'],0).success);
    assert.equal(after.boardIndustries.nuneaton_0.resourceCubes,0);
    for(const s of [before,after])s.endCanalEra();
    const connections=['birmingham-coventry','coventry-nuneaton'];
    before.boardLinks['birmingham-oxford']={playerId:1,type:'rail'};
    after.boardLinks['birmingham-oxford']={playerId:1,type:'rail'};
    assert.ok(new Logic(before).planNetwork(1,connections));
    assert.equal(new Logic(after).planNetwork(1,connections),null);
});
test('double-rail financing requires coal purchases in addition to the printed £15',()=>{
    const s=beerRace();s.endCanalEra();s.boardLinks['birmingham-oxford']={playerId:1,type:'rail'};
    s.players[1].money=15;
    assert.equal(new Logic(s).planNetwork(1,['birmingham-coventry','coventry-nuneaton']),null);
    s.players[1].money=90;
    const plan=new Logic(s).planNetwork(1,['birmingham-coventry','coventry-nuneaton']);assert.ok(plan);assert.ok(plan.cost>15);
});
test('merchant availability changes legal industry routes rather than prescribing a fixed opening',()=>{
    const s=S.create(4,312010023);s.players[0].money=100;
    s.players[0].hand=[{type:'location',location:'birmingham'}];
    s.boardLinks['birmingham-oxford']={playerId:0,type:'canal'};
    s.boardIndustries.birmingham_0={playerId:0,type:'manufacturer',tileData:{...D.INDUSTRY_DATA.manufacturer[0],beersToSell:0},resourceCubes:0,flipped:false};
    s.merchantTiles=[{location:'oxford',buys:'cottonMill',hasBeer:true}];
    assert.equal(new Logic(s).planSales(0,['birmingham_0']),null);
    s.merchantTiles[0].buys='manufacturer';assert.ok(new Logic(s).planSales(0,['birmingham_0']));
});
test('retained-card candidates protect usable pottery access over an exhausted industry',()=>{
    const s=S.create(4,312010024);s.currentPlayerIndex=s.turnOrder.indexOf(0);
    s.players[0].hand=[{type:'industry',industryType:'pottery'},{type:'industry',industryType:'coalMine'}];
    s.players[0].industryTiles.coalMine.forEach(t=>t.used=true);
    const normal=S.candidates(s).find(a=>a.action==='loan');assert.equal(normal.cardIndex,0);
    const retained=S.candidates(s,{retainCards:true}).find(a=>a.action==='loan');assert.equal(retained.cardIndex,1);
    const next=S.step(s,retained).state;assert.equal(next.players[0].hand[0].industryType,'pottery');
});
