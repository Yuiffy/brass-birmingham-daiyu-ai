const test=require('node:test'),assert=require('node:assert/strict');
const {grouped,interval,metrics,analyze}=require('../world_model/human_analysis');
test('statistics resample seed groups, keeping correlated seats and rotations together',()=>{
 const report={games:[0,1].flatMap(group=>[0,1,2,3].map(rotation=>({group,rotation,scores:
     Array.from({length:4},()=>({profile:'human',vp:group?160:80}))})))};
 assert.deepEqual(grouped(report,'human',s=>s.vp),[80,160]);
 const result=metrics(report,'human');assert.equal(result.independentGroups,2);
 assert.equal(result.meanVP,120);assert.equal(result.atLeast140,.5);
 assert.deepEqual(result.meanVP95CI,[80,160]);
 assert.deepEqual(interval([7,7,7]),[7,7]);
});
test('formal analysis rejects incompatible or incomplete benchmark reports',()=>{
 assert.throws(()=>analyze({completedGames:8},{},{}),/120/);
 const mock={completedGames:120,independentSeedGroups:30,modelSHA256:'a',sourceSHA256:{s:'x'},config:{depth:2,width:8,weight:.8}};
 assert.throws(()=>analyze(mock,{...mock,modelSHA256:'b'},mock),/budget|frozen/);
});
