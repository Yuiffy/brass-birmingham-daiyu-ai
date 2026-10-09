// Compare strategy profiles using the same engine and immutable value weights.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const {Worker,isMainThread,parentPort,workerData}=require('node:worker_threads');
process.env.BRASS_RULES='economy-v2';
const Sim=require('./simulator'),Planner=require('./planner'),{Network}=require('./inference');
const profiles={baseline:{},double:{strategy:'double-rail-v1'},human:{strategy:'human-guide-v1'},chain:{strategy:'human-chain-v2'},finance:{strategy:'human-finance-v2'},cards:{strategy:'human-card-v2'},
    trained:{strategy:'human-guide-v1'},trainedchain:{strategy:'human-chain-v2'},trainedcards:{strategy:'human-card-v2'},world:{type:'world',doubleRail:true},oldworld:{type:'world'}};

function runGroup(config,group) {
    const world=new Network(JSON.parse(fs.readFileSync(config.model,'utf8')));
    const candidate=config.candidateModel?new Network(JSON.parse(fs.readFileSync(config.candidateModel,'utf8'))):world;
    const players=config.lineup.length,records=[];
    for(let rotation=0;rotation<players;rotation++) {
        const seats=config.lineup.map((_,p)=>config.lineup[(p+rotation)%players]);
        const seed=config.seed+group*9973;
        let state=Sim.create(players,seed,seats),moves=0;
        const details=seats.map(()=>({canalIndustry:0,canalLinks:0,railIndustry:0,railLinks:0,
            actions:{},doubleRails:0,thinkingMs:0,trace:[]}));
        while(!state.gameOver) {
            const actor=state.currentPlayerId,profile=seats[actor],d=details[actor],era=state.era;
            const selectedWorld=['trained','trainedchain','trainedcards','world'].includes(profile)?candidate:world;
            const result=Planner.plan(state,{type:'guided',world:selectedWorld,depth:config.depth,width:config.width,
                knowledgeWeight:config.weight,...profiles[profile]});
            if(!result.selected)throw Error('No selected action');
            const a=result.selected;d.thinkingMs+=result.elapsedMs;
            d.actions[a.action]=(d.actions[a.action]||0)+1;if(a.target?.connectionIds)d.doubleRails++;
            d.trace.push({era,round:state.round,action:Sim.label(a),money:state.currentPlayer.money,
                income:state.getIncomeAmount(state.currentPlayer.income)});
            const next=Sim.step(state,a);state=next.state;
            for(const score of next.event.scores||[]) {
                details[score.playerId][era+'Industry']+=score.industryVP;
                details[score.playerId][era+'Links']+=score.linkVP;
            }
            if(++moves>1000)throw Error('Game did not terminate');
        }
        const best=state.players.slice().sort((a,b)=>state.comparePlayers(a,b))[0];
        const winners=state.players.filter(p=>state.comparePlayers(p,best)===0);
        const scores=state.players.map((p,seat)=>({seat,profile:seats[seat],vp:p.vp,money:p.money,
            income:state.getIncomeAmount(p.income),winShare:winners.some(w=>w.id===seat)?1/winners.length:0,
            ...details[seat]}));
        records.push({group,rotation,seed,seats,moves,scores});
    }
    return records;
}

function summarize(records) {
    const names=[...new Set(records.flatMap(g=>g.seats))];
    return names.map(profile=>{
        const entries=records.flatMap(g=>g.scores.filter(s=>s.profile===profile));
        const sorted=entries.map(s=>s.vp).sort((a,b)=>a-b),sum=k=>entries.reduce((n,s)=>n+s[k],0);
        const actions={};for(const s of entries)for(const [k,n] of Object.entries(s.actions))actions[k]=(actions[k]||0)+n/entries.length;
        const moves=entries.reduce((n,s)=>n+Object.values(s.actions).reduce((a,b)=>a+b,0),0);
        return {profile,appearances:entries.length,meanVP:sum('vp')/entries.length,
            medianVP:sorted[Math.floor(sorted.length/2)],p10:sorted[Math.floor(sorted.length*.1)],
            p90:sorted[Math.min(sorted.length-1,Math.floor(sorted.length*.9))],minVP:sorted[0],maxVP:sorted.at(-1),
            under100:entries.filter(s=>s.vp<100).length,atLeast140:entries.filter(s=>s.vp>=140).length/entries.length,
            atLeast150:entries.filter(s=>s.vp>=150).length/entries.length,winShare:sum('winShare')/entries.length,
            averageMoney:sum('money')/entries.length,doubleRails:sum('doubleRails')/entries.length,
            millisecondsPerMove:sum('thinkingMs')/moves,meanActions:actions,
            canalIndustry:sum('canalIndustry')/entries.length,canalLinks:sum('canalLinks')/entries.length,
            railIndustry:sum('railIndustry')/entries.length,railLinks:sum('railLinks')/entries.length};
    });
}

async function benchmark(config) {
    const players=config.lineup.length;
    if(players<2||players>4||config.lineup.some(p=>!profiles[p])||config.games%players||config.games<players||
        !Number.isInteger(config.workers)||config.workers<1||config.workers>16)throw Error('Use complete rotations, 2–4 seats and 1–16 workers');
    const sha=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
    const sources=['human_strategy.js','production_chain.js','planner.js','simulator.js','encoding.js','value_features.js','inference.js',
        '../js/gameLogic.js','../js/gameState.js','../js/gameData.js','../scripts/autorun.js','human_benchmark.js'];
    const sourceSHA256=Object.fromEntries(sources.map(f=>[f,sha(path.join(__dirname,f))]));
    const modelSHA256=sha(config.model);
    const candidateModelSHA256=config.candidateModel?sha(config.candidateModel):null;
    const groups=config.games/players,count=Math.min(config.workers,groups),threads=[];
    let completed=0;
    try {
        const work=Array.from({length:count},(_,i)=>Array.from({length:groups},(_,g)=>g).filter(g=>g%count===i));
        const output=await Promise.all(work.map(groupIds=>new Promise((resolve,reject)=>{
            const worker=new Worker(__filename,{workerData:{config,groups:groupIds}});threads.push(worker);let done=false;
            worker.on('message',m=>{
                if(m.progress){completed++;console.log(`Completed seed groups ${completed}/${groups}`);}
                else {done=true;resolve(m.records);}
            });
            worker.on('error',reject);worker.on('exit',code=>{if(!done)reject(Error('Worker exited without records: '+code));});
        })));
        const records=output.flat().sort((a,b)=>a.group-b.group||a.rotation-b.rotation);
        if(sources.some(f=>sourceSHA256[f]!==sha(path.join(__dirname,f)))||modelSHA256!==sha(config.model)||candidateModelSHA256&&candidateModelSHA256!==sha(config.candidateModel))
            throw Error('Source or model changed during the benchmark; discard this run');
        return {generatedAt:new Date().toISOString(),config,rules:'economy-v2-network-corrected',
            sourceSHA256,modelSHA256,candidateModelSHA256,independentSeedGroups:groups,completedGames:records.length,
            rows:summarize(records),games:records,
            notes:['All scores use the same calibrated engine without terminal income bonuses.',
                'Baseline weights are immutable. One complete seat rotation stays within each seed group.',
                'Human profile blends learned value with source-derived strategy estimates; no new neural training is implied.',
                'These scores do not certify tabletop rules or human-level skill.']};
    } finally {await Promise.all(threads.map(w=>w.terminate()));}
}

if(!isMainThread) {
    const records=[];
    for(const group of workerData.groups){records.push(...runGroup(workerData.config,group));parentPort.postMessage({progress:true});}
    parentPort.postMessage({records});parentPort.close();
} else if(require.main===module) {
    const options={};const args=process.argv.slice(2);
    for(let i=0;i<args.length;i+=2)options[args[i].replace(/^--/,'')]=args[i+1];
    const players=Number(options.players||4);
    const config={games:Number(options.games||8),seed:Number(options.seed||209100001),depth:Number(options.depth||2),
        width:Number(options.width||8),weight:Number(options.weight||.8),workers:Number(options.workers||4),
        lineup:(options.lineup||'human,baseline,baseline,baseline').split(','),
        model:path.resolve(options.models||`world_model/experiments/economy-20260920/${players}p/direct/world-model.json`),
        candidateModel:options.candidate?path.resolve(options.candidate):null};
    benchmark(config).then(report=>{
        const out=options.out||'world_model/.local-experiments/human-guide/trial.json';
        fs.mkdirSync(path.dirname(out),{recursive:true});fs.writeFileSync(out,JSON.stringify(report,null,2));
        console.table(report.rows);console.log('Saved '+out);
    }).catch(e=>{console.error(e);process.exitCode=1;});
}
module.exports={runGroup,summarize,benchmark};
