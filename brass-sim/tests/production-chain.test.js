const test=require('node:test'),assert=require('node:assert/strict');
process.env.BRASS_RULES='economy-v2';
const S=require('../world_model/simulator'),C=require('../world_model/production_chain');
function position(money=12) {
    const s=S.create(4,312010001);s.players[0].money=money;
    s.players[0].hand=[{type:'location',location:'birmingham'},...Array.from({length:5},()=>({type:'industry',industryType:'ironWorks'}))];
    s.merchantTiles=[{location:'oxford',buys:'cottonMill',hasBeer:true,bonusClaimed:false}];
    return s;
}
const goal={type:'cottonMill',city:'birmingham',slot:0,develop:0};
test('chain finances a route with a real loan action and income reduction',()=>{
    const s=position(),before=S.snapshot(s),p=C.project(s,0,goal);
    assert.ok(p);assert.equal(p.loans,1);assert.equal(p.actions,4);
    assert.equal(p.spending,15);assert.equal(p.trace.find(x=>x.action==='loan').income,-3);
    assert.equal(p.trace.at(-1).action,'sell');assert.deepEqual(S.snapshot(s),before);
});
test('cash reserve removes the loan and saves an action',()=>{
    const p=C.project(position(30),0,goal);assert.ok(p);assert.equal(p.loans,0);assert.equal(p.actions,3);
});
test('two remaining cards cannot finance, connect and sell an industry',()=>{
    const s=position();s.players[0].hand=s.players[0].hand.slice(0,2);s.drawDeck=[];
    assert.equal(C.project(s,0,goal),null);
});
test('low-income player cannot obtain unlimited hypothetical loans',()=>{
    const s=position(0);s.players[0].income=0;
    // Position zero is -10 income in the calibrated track.
    assert.equal(s.getIncomeAmount(s.players[0].income),-10);
    assert.equal(C.project(s,0,goal),null);
});
test('no beer and no brewery supply rejects an unrealizable sale',()=>{
    const s=position(100);s.merchantTiles[0].hasBeer=false;
    s.players[0].industryTiles.brewery.forEach(t=>t.used=true);
    assert.equal(C.project(s,0,goal),null);
});
test('a critical city card is retained for build while paying loan',()=>{
    const p=C.project(position(0),0,goal);assert.ok(p);
    assert.equal(p.trace[0].action,'loan');assert.equal(p.trace[0].card.industryType,'ironWorks');
    assert.equal(p.trace.find(x=>x.action==='build').card.location,'birmingham');
});
test('chain planning never depends on real hidden deck order',()=>{
    const s=position(30),t=S.clone(s);t.drawDeck.reverse();t._rngState=44;
    assert.deepEqual(C.plans(s,0),C.plans(t,0));
});
