// Measure exactly the AI options available in browser setup, by player count.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),zlib=require('node:zlib');
const {Worker,isMainThread,parentPort,workerData}=require('node:worker_threads');
const TYPES=['heuristic','search','neural','world','guided'];
const sha=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
const read=file=>JSON.parse(fs.readFileSync(file,'utf8'));
function configurations(){
    const entries=[];
    for(const rules of ['legacy-v1','economy-v2'])for(const players of [2,3,4])
        for(const profile of rules==='economy-v2'?['','human-guide-v1','teacher-trained-v2']:[''])for(const type of TYPES){
            // These profiles only change guided strategy or teacher world weights.
            // Teacher 4P also enables double rails in plain search via model metadata.
            const effective=type==='guided'?profile:type==='world'&&players===4&&profile==='teacher-trained-v2'?profile:
                type==='search'&&players===4&&profile==='teacher-trained-v2'?profile:'';
            entries.push({rules,players,profile,type,key:`${rules}/${players}/${type}/${effective||'standard'}`,effective});
        }
    return entries;
}
function interval(values,seed=49183){
    let r=seed|0;const random=()=>{r^=r<<13;r^=r>>>17;r^=r<<5;return (r>>>0)/4294967296;};
    const means=[];
    for(let i=0;i<4000;i++){let sum=0;for(let n=0;n<values.length;n++)sum+=values[Math.floor(random()*values.length)];means.push(sum/values.length);}
    means.sort((a,b)=>a-b);return [means[100],means[3899]];
}
function summarize(games){
    const scores=games.flatMap(g=>g.scores.map(s=>s.vp)).sort((a,b)=>a-b);
    const means=games.map(g=>g.scores.reduce((n,s)=>n+s.vp,0)/g.scores.length);
    return {games:games.length,appearances:scores.length,meanVP:means.reduce((a,b)=>a+b,0)/means.length,
        meanVP95CI:interval(means),minVP:scores[0],maxVP:scores.at(-1),
        medianVP:scores[Math.floor(scores.length/2)],p10:scores[Math.floor(scores.length*.1)],
        under100:scores.filter(v=>v<100).length/scores.length,atLeast140:scores.filter(v=>v>=140).length/scores.length,
        atLeast150:scores.filter(v=>v>=150).length/scores.length,
        millisecondsPerMove:games.reduce((n,g)=>n+g.thinkingMs,0)/games.reduce((n,g)=>n+g.trace.length,0)};
}
function replay(Sim,games,players,type){
    let actions=0;
    for(const game of games){
        let state=Sim.create(players,game.seed,Array(players).fill(type));
        for(const t of game.trace){
            if(state.currentPlayerId!==t.actor||state.era!==t.era||state.round!==t.round)throw Error('Replay turn mismatch');
            state=Sim.step(state,t.action).state;actions++;
        }
        if(!state.gameOver)throw Error('Incomplete replay');
        for(const s of game.scores){const p=state.players[s.seat];
            if(p.vp!==s.vp||p.money!==s.money||p.income!==s.income)throw Error('Replay terminal mismatch');
        }
    }
    return {games:games.length,actions,legalAndTerminal:true,allScoresCashIncomeMatch:true};
}
async function runJob(config,protocol){
    process.env.BRASS_RULES=config.rules;
    const Sim=require('./simulator'),P=require('./planner'),Runtime=require('./runtime_models'),{Network}=require('./inference');
    const directory=path.join(__dirname,Runtime.directoryForPlayers(config.players));
    const base=read(path.join(directory,'world-model.json'));
    const data=config.effective==='teacher-trained-v2'?Runtime.teacherDataForPlayers(config.players,base,
        read(path.join(__dirname,'experiments/human-teacher-20261009/models/world-model.json'))):base;
    const world=new Network(data),policyDirectory=path.join(__dirname,Runtime.directoryForPlayers(config.rules==='economy-v2'?config.players:4));
    const policy=new Network(read(path.join(policyDirectory,'neural-policy.json')));
    const planOptions={type:config.type,world,policy,depth:2,width:8,
        strategy:config.type==='guided'?Runtime.strategyForProfile(config.effective):null};
    const games=[];
    for(let group=0;group<protocol.gamesPerConfig;group++){
        const seed=protocol.seed+config.players*100000+group*9973;
        let state=Sim.create(config.players,seed,Array(config.players).fill(config.type));const trace=[];let thinkingMs=0;
        while(!state.gameOver){
            const result=P.plan(state,planOptions);thinkingMs+=result.elapsedMs;
            if(!result.selected)throw Error('No action');
            trace.push({actor:state.currentPlayerId,era:state.era,round:state.round,action:structuredClone(result.selected)});
            state=Sim.step(state,result.selected).state;
            if(trace.length>1000)throw Error('Game failed to terminate');
        }
        games.push({group,seed,thinkingMs,trace,scores:state.players.map(p=>({seat:p.id,vp:p.vp,money:p.money,income:p.income}))});
        parentPort?.postMessage({progress:group+1});
    }
    // A separate pass executes saved actions without consulting any planner/model.
    return {config,protocolSHA256:sha(JSON.stringify(protocol)),worldSHA256:sha(JSON.stringify(data)),
        policySHA256:sha(JSON.stringify(policy.data)),summary:summarize(games),audit:replay(Sim,games,config.players,config.type),games};
}
function fileName(key){return key.replaceAll('/','-')+'.json.gz';}
async function benchmark({out,gamesPerConfig=32,seed=620260101,workers=16}={}){
    if(!out||!Number.isInteger(gamesPerConfig)||gamesPerConfig<2||!Number.isInteger(workers)||workers<1||workers>16)throw Error('Invalid benchmark configuration');
    fs.mkdirSync(out,{recursive:true});
    const entries=configurations(),jobs=[...new Map(entries.map(e=>[e.key,e])).values()];
    const sources=['browser_benchmark.js','runtime_models.js','planner.js','human_strategy.js','production_chain.js','human_intent.js',
        'simulator.js','encoding.js','value_features.js','inference.js','../js/gameData.js','../js/gameState.js','../js/gameLogic.js','../scripts/autorun.js'];
    const modelFiles=['models/world-model.json','models/neural-policy.json','experiments/human-teacher-20261009/models/world-model.json',
        ...[2,3].flatMap(p=>[`experiments/dynamics-${p}p-active/models/world-model.json`]),
        ...[2,3,4].flatMap(p=>[`experiments/economy-20260920/${p}p/direct/world-model.json`,`experiments/economy-20260920/${p}p/direct/neural-policy.json`])];
    const bytes=Object.fromEntries([...sources,...modelFiles].map(f=>[f,sha(fs.readFileSync(path.join(__dirname,f)))]));
    const protocol={version:1,gamesPerConfig,seed,depth:2,width:8,entries,jobs,sourceAndModelSHA256:bytes,
        methodology:'All seats use the same AI. Each game has a distinct seed; identical-policy seat rotations are not duplicated. Bootstrap resamples complete games, never individual seats.',
        transfer:'2P/3P guided teacher uses the frozen 4P value head and card-retention strategy with real-engine transitions. Native 2P/3P world agents remain unchanged. No additional neural training.',
        notes:['Legacy and economy scores are incomparable. Self-play scores do not predict scores against humans or certify official rules.','Identical runtime options share one measured result; every browser combination has an explicit mapping.']};
    fs.writeFileSync(path.join(out,'protocol.json'),JSON.stringify(protocol,null,2));
    const hash=sha(JSON.stringify(protocol)),results=new Map();let next=0,finished=0;
    const consume=async()=>{
        while(next<jobs.length){
            const job=jobs[next++],file=path.join(out,fileName(job.key));let report;
            if(fs.existsSync(file)){
                report=JSON.parse(zlib.gunzipSync(fs.readFileSync(file)));
                if(report.protocolSHA256!==hash)throw Error('Existing results have a different frozen protocol');
            }else report=await new Promise((resolve,reject)=>{
                const w=new Worker(__filename,{workerData:{job,protocol}});let received=false;
                w.on('message',m=>{if(m.report){received=true;resolve(m.report);}else if(m.progress%8===0)console.log(`${job.key}: ${m.progress}/${gamesPerConfig} games`);});
                w.on('error',reject);w.on('exit',code=>{if(!received)reject(Error('Worker exited without report: '+code));});
            });
            if(Object.entries(bytes).some(([f,h])=>sha(fs.readFileSync(path.join(__dirname,f)))!==h))throw Error('Execution source/model changed; discard results');
            fs.writeFileSync(file,zlib.gzipSync(JSON.stringify(report)));
            results.set(job.key,report);console.log(`Finished ${++finished}/${jobs.length}: ${job.key}, mean ${report.summary.meanVP.toFixed(2)}`);
        }
    };
    await Promise.all(Array.from({length:Math.min(workers,jobs.length)},consume));
    const scores=entries.map(e=>({...e,...results.get(e.key).summary,report:fileName(e.key)}));
    const summary={generatedAt:new Date().toISOString(),protocolSHA256:hash,methodology:protocol.methodology,transfer:protocol.transfer,
        uniqueConfigurations:jobs.length,browserCombinations:entries.length,
        totalGames:[...results.values()].reduce((n,r)=>n+r.games.length,0),
        replayedActions:[...results.values()].reduce((n,r)=>n+r.audit.actions,0),allReplayAuditsPassed:true,scores};
    fs.writeFileSync(path.join(out,'summary.json'),JSON.stringify(summary,null,2));
    fs.writeFileSync(path.join(out,'browser-ai-scores.js'),'globalThis.BrassBenchmarkScores = '+JSON.stringify(summary)+';\n');
    console.log('All browser configurations measured and replay-audited. '+out);
    return summary;
}
if(!isMainThread){runJob(workerData.job,workerData.protocol).then(report=>{parentPort.postMessage({report});parentPort.close();}).catch(e=>{throw e;});}
else if(require.main===module){
    const args={};for(let i=2;i<process.argv.length;i+=2)args[process.argv[i].replace(/^--/,'')]=process.argv[i+1];
    benchmark({out:path.resolve(args.out||'world_model/.local-experiments/browser-all-20261010'),gamesPerConfig:Number(args.games||32),
        seed:Number(args.seed||620260101),workers:Number(args.workers||16)}).catch(e=>{console.error(e);process.exitCode=1;});
}
module.exports={configurations,summarize,replay,benchmark};
