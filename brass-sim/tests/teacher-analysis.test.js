const test=require('node:test'),assert=require('node:assert/strict');
const {analyze}=require('../world_model/analyze_teacher_league');
function report(lineup,gain=0){return {completedGames:120,independentSeedGroups:30,modelSHA256:'old',candidateModelSHA256:'new',sourceSHA256:{code:'frozen'},config:{depth:2,width:8,weight:.8},
 games:Array.from({length:30},(_,group)=>Array.from({length:4},(_,rotation)=>({group,rotation,seed:123+group,
 scores:lineup.map((profile,seat)=>({profile,seat,vp:120+group+(profile==='trainedcards'?gain:0)}))}))).flat()};}
test('teacher analysis pairs seat and seed and is insensitive to self-play record order',()=>{
 const baseline=report(Array(4).fill('human')),mixed=report(['trainedcards','human','human','human'],3),self=report(Array(4).fill('trainedcards'),2);
 const a=analyze(mixed,baseline,self);assert.equal(a.pairedGain.meanVP,3);assert.equal(a.selfPlayGain.meanVP,2);
 self.games.reverse();assert.deepEqual(analyze(mixed,baseline,self).selfPlayGain,a.selfPlayGain);
 self.games[0].seed++;assert.throws(()=>analyze(mixed,baseline,self),/Unpaired/);
});
test('teacher analysis rejects duplicate records and artifact mismatch',()=>{
 const baseline=report(Array(4).fill('human')),mixed=report(['trainedcards','human','human','human']),self=report(Array(4).fill('trainedcards'));
 assert.throws(()=>analyze(mixed,baseline,{...self,candidateModelSHA256:'other'}),/artifacts/);
 mixed.games[1]=mixed.games[0];assert.throws(()=>analyze(mixed,baseline,self),/duplicate/);
});
