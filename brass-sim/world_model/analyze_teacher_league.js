const fs=require('node:fs'),path=require('node:path');
const {average,grouped,interval,metrics}=require('./human_analysis');
function analyze(mixed,baseline,selfplay,{candidate='trainedcards',incumbent='human'}={}) {
    for(const report of [mixed,baseline,selfplay]) {
        if(report.completedGames!==120||report.independentSeedGroups!==30)throw Error('Need 30 complete seed groups / 120 games per setting');
        if(report.modelSHA256!==mixed.modelSHA256||report.candidateModelSHA256!==mixed.candidateModelSHA256||
            JSON.stringify(report.sourceSHA256)!==JSON.stringify(mixed.sourceSHA256))throw Error('Frozen artifacts differ');
        for(const name of ['depth','width','weight'])if(report.config[name]!==mixed.config[name])throw Error('Unequal planning configuration');
    }
    const refs=new Map(baseline.games.map(g=>[`${g.group}:${g.rotation}`,g]));
    if(refs.size!==120)throw Error('Duplicate baseline rotations');
    for(const report of [mixed,selfplay]) {
        const keys=new Set();
        for(const g of report.games) {
            const key=`${g.group}:${g.rotation}`,ref=refs.get(key);
            if(keys.has(key)||!ref||ref.seed!==g.seed)throw Error('Unpaired or duplicate rotations');
            keys.add(key);
        }
        if(keys.size!==120)throw Error('Missing game records');
    }
    const delta=new Map();
    for(const g of mixed.games) {
        const ref=refs.get(`${g.group}:${g.rotation}`),own=g.scores.filter(s=>s.profile===candidate);
        if(!ref||g.seed!==ref.seed||own.length!==1)throw Error('Unpaired games');
        const other=ref.scores.find(s=>s.seat===own[0].seat);
        if(other.profile!==incumbent)throw Error('Wrong incumbent');
        if(!delta.has(g.group))delta.set(g.group,[]);delta.get(g.group).push(own[0].vp-other.vp);
    }
    const gain=[...delta.values()].map(average),selfDelta=new Map();
    for(const g of selfplay.games) {
        const ref=refs.get(`${g.group}:${g.rotation}`);
        if(g.scores.some(s=>s.profile!==candidate)||ref.scores.some(s=>s.profile!==incumbent))throw Error('Wrong self-play lineup');
        if(!selfDelta.has(g.group))selfDelta.set(g.group,[]);
        selfDelta.get(g.group).push(average(g.scores.map(s=>s.vp))-average(ref.scores.map(s=>s.vp)));
    }
    const selfGain=[...selfDelta.values()].map(average);
    const fixed=metrics(mixed,candidate),self=metrics(selfplay,candidate),reference=metrics(baseline,incumbent);
    const tail=report=>grouped(report,report===baseline?incumbent:candidate,s=>+(s.vp<100));
    return {generatedAt:new Date().toISOString(),candidate,incumbent,modelSHA256:mixed.modelSHA256,candidateModelSHA256:mixed.candidateModelSHA256,
        sourceSHA256:mixed.sourceSHA256,fixedOpponents:fixed,candidateSelfPlay:self,incumbentSelfPlay:reference,
        pairedGain:{meanVP:average(gain),meanVP95CI:interval(gain)},selfPlayGain:{meanVP:average(selfGain),meanVP95CI:interval(selfGain)},
        under100:{mixed:average(tail(mixed)),selfplay:average(tail(selfplay)),incumbent:average(tail(baseline))},
        supported:{improvementOverIncumbent:interval(gain)[0]>0,
            selfPlayImprovement:interval(selfGain)[0]>0,
            tailNotWorse:average(tail(selfplay))<=average(tail(baseline)),
            stable140:[fixed,self].every(m=>m.meanVP95CI[0]>=140&&m.atLeast14095CI[0]>=.8),
            stable150:[fixed,self].every(m=>m.meanVP95CI[0]>=150&&m.atLeast15095CI[0]>=.8)},
        notes:['Whole independent seed groups bootstrap; identical self-play rotations are correlated.',
            'Current human-guide-v1 is the opponent and reference, not the weak historical guided agent.',
            'All hands public; synthetic teacher training does not establish human match strength.']};
}
if(require.main===module){const root=process.argv[2];const read=n=>{const p=path.join(root,n+'.json');return JSON.parse(fs.existsSync(p)?fs.readFileSync(p):require('zlib').gunzipSync(fs.readFileSync(p+'.gz')));};
    const report=analyze(read('mixed'),read('baseline'),read('selfplay'));fs.writeFileSync(path.join(root,'summary.json'),JSON.stringify(report,null,2));console.log(report);}
module.exports={analyze};
