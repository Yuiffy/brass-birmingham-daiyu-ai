// Independently replay recorded actions without invoking any planner or model.
const fs=require('node:fs'),path=require('node:path'),zlib=require('node:zlib'),assert=require('node:assert/strict');
process.env.BRASS_RULES='economy-v2';
const S=require('./simulator'),D=require('./settlement_diagnostics');
function audit(root){
 const cache=new Map();let games=0,moves=0;
 for(const name of ['baseline','mixed','selfplay','world']){
  const file=path.join(root,name+'.json'),report=JSON.parse(fs.existsSync(file)?fs.readFileSync(file):zlib.gunzipSync(fs.readFileSync(file+'.gz')));
  for(const game of report.games){
   const trace=game.scores.flatMap(s=>s.trace.map(t=>({actor:s.seat,...t}))).sort((a,b)=>a.seq-b.seq);
   const signature=JSON.stringify([game.seed,game.seats]),record=JSON.stringify({trace,scores:game.scores.map(s=>({vp:s.vp,money:s.money,income:s.income,settlements:s.settlements}))});
   if(cache.has(signature)){assert.equal(record,cache.get(signature));games++;continue;}
   let state=S.create(game.seats.length,game.seed,game.seats);const settlements=game.seats.map(()=>[]);
   trace.forEach((t,i)=>{
    assert.equal(t.seq,i);assert.equal(t.actor,state.currentPlayerId);assert.equal(t.era,state.era);assert.equal(t.round,state.round);
    assert.equal(t.money,state.currentPlayer.money);assert.equal(t.income,state.getIncomeAmount(state.currentPlayer.income));
    const next=S.step(state,t.rawAction,{captureSettlement:true});state=next.state;
    if(next.event.beforeSettlement){const before=Object.assign(S.clone(state),next.event.beforeSettlement);for(const r of D.settlement(before))settlements[r.player].push(r);}
    moves++;
   });
   assert.ok(state.gameOver);assert.equal(trace.length,game.moves);
   for(const score of game.scores){const p=state.players[score.seat];assert.equal(p.vp,score.vp);assert.equal(p.money,score.money);assert.equal(state.getIncomeAmount(p.income),score.income);assert.deepEqual(settlements[score.seat],score.settlements);}
   cache.set(signature,record);games++;
  }
 }
 return {games,distinctSeedLineups:cache.size,replayedActions:moves,allTerminalScoresAndSettlementsMatch:true,
  duplicatePolicy:'Identical seed and lineup records must have identical complete actions and outcomes; replay once.'};
}
if(require.main===module){const result=audit(process.argv[2]);fs.writeFileSync(path.join(process.argv[2],'replay-audit.json'),JSON.stringify(result,null,2));console.log(result);}
module.exports={audit};
