// Statistics use independent seed groups, never seats as independent samples.
const fs=require('node:fs'),path=require('node:path');
function average(values){return values.reduce((a,b)=>a+b,0)/values.length;}
function grouped(report,profile,metric){
    const groups=new Map();
    for(const game of report.games)for(const score of game.scores.filter(s=>s.profile===profile)){
        if(!groups.has(game.group))groups.set(game.group,[]);
        groups.get(game.group).push(metric(score));
    }
    return [...groups.values()].map(average);
}
function interval(groups,draws=10000,seed=81009){
    if(!groups.length||groups.some(v=>!Number.isFinite(v)))throw Error('Invalid independent groups');
    let x=seed>>>0;
    const random=()=>{x=(Math.imul(1664525,x)+1013904223)>>>0;return x/4294967296;};
    const samples=Array.from({length:draws},()=>average(groups.map(()=>groups[Math.floor(random()*groups.length)]))).sort((a,b)=>a-b);
    return [samples[Math.floor(draws*.025)],samples[Math.min(draws-1,Math.floor(draws*.975))]];
}
function metrics(report,profile){
    const vp=grouped(report,profile,s=>s.vp),p140=grouped(report,profile,s=>+(s.vp>=140)),p150=grouped(report,profile,s=>+(s.vp>=150));
    return {independentGroups:vp.length,meanVP:average(vp),meanVP95CI:interval(vp),
        atLeast140:average(p140),atLeast14095CI:interval(p140),atLeast150:average(p150),atLeast15095CI:interval(p150)};
}
function analyze(mixed,baseline,selfplay){
    for(const report of [mixed,baseline,selfplay]){
        if(report.completedGames!==120||report.independentSeedGroups!==30)throw Error('Expected 120 complete games / 30 seed groups');
        if(report.modelSHA256!==mixed.modelSHA256||JSON.stringify(report.sourceSHA256)!==JSON.stringify(mixed.sourceSHA256))throw Error('Mismatched frozen source/model');
        if(report.config.depth!==2||report.config.width!==8||report.config.weight!==.8)throw Error('Mismatched planning budget');
    }
    const differences=new Map(),baselineSeats=new Map(baseline.games.map(g=>[`${g.group}:${g.rotation}`,g]));
    for(const game of mixed.games){
        const reference=baselineSeats.get(`${game.group}:${game.rotation}`);
        if(reference?.seed!==game.seed)throw Error('Unpaired seed or rotation');
        const human=game.scores.filter(s=>s.profile==='human');
        if(human.length!==1)throw Error('Expected one candidate per mixed game');
        const own=human[0],other=reference.scores.find(s=>s.seat===own.seat);
        if(other?.profile!=='baseline')throw Error('Invalid frozen opponent');
        if(!differences.has(game.group))differences.set(game.group,[]);
        differences.get(game.group).push(own.vp-other.vp);
    }
    const delta=[...differences.values()].map(average),fixed=metrics(mixed,'human'),self=metrics(selfplay,'human');
    return {generatedAt:new Date().toISOString(),modelSHA256:mixed.modelSHA256,sourceSHA256:mixed.sourceSHA256,
        fixedOpponents:fixed,baselineSelfPlay:metrics(baseline,'baseline'),candidateSelfPlay:self,
        pairedGain:{independentGroups:delta.length,meanVP:average(delta),meanVP95CI:interval(delta)},
        supported:{improvementOverBaseline:interval(delta)[0]>0,
            stable140:[fixed,self].every(m=>m.meanVP95CI[0]>=140&&m.atLeast14095CI[0]>=.8),
            stable150:[fixed,self].every(m=>m.meanVP95CI[0]>=150&&m.atLeast15095CI[0]>=.8)},
        deployment:'explicit experimental option; original guided default and weights retained',
        notes:['95% percentile bootstrap, 10000 deterministic draws of whole seed groups.',
            'Stable means at least 80% of appearances above the threshold in both tested environments.',
            'Self-play rotations with identical profiles can repeat games; groups remain the sampling unit.',
            'These intervals concern game-seed variability with one frozen model, not human-level skill or training-seed variability.',
            'Different root and continuation pools make this a strategy-package comparison, not equal-compute architecture superiority.']};
}
if(require.main===module){
    const directory=path.resolve(process.argv[2]||'world_model/experiments/human-guide-20261009');
    const read=file=>{
        const filename=path.join(directory,file);
        return JSON.parse(fs.existsSync(filename)?fs.readFileSync(filename,'utf8'):
            require('node:zlib').gunzipSync(fs.readFileSync(filename+'.gz')).toString('utf8'));
    };
    const result=analyze(read('mixed.json'),read('baseline.json'),read('selfplay.json'));
    fs.writeFileSync(path.join(directory,'summary.json'),JSON.stringify(result,null,2));
    console.log(JSON.stringify(result,null,2));
}
module.exports={average,grouped,interval,metrics,analyze};
