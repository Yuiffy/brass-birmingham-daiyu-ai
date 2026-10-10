// Synthetic complete games from named teachers, never labeled human replays.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const {Worker,isMainThread,parentPort,workerData}=require('node:worker_threads');
const Sim=require('./simulator'),E=require('./encoding'),P=require('./planner'),{Network}=require('./inference');

function shard(config){
    const world=new Network(JSON.parse(fs.readFileSync(path.join(config.models,'world-model.json'))));
    const policy=new Network(JSON.parse(fs.readFileSync(path.join(config.models,'neural-policy.json'))));
    const fd=fs.openSync(config.file,'wx'),games=[];let rows=0;
    const transitionFd=config.transitionFile?fs.openSync(config.transitionFile,'wx'):null;
    const transitionGames=[],counts={};let transitionRows=0;
    const players=config.players||4;
    const configuredLineup=config.lineup?.length?config.lineup:null;
    if(configuredLineup&&configuredLineup.length<players)throw Error('Replay lineup must include at least as many entrants as players');
    try{for(let g=config.start;g<config.end;g++){
        const seed=config.seed+g*9973,rng=Sim.rng(seed^0x58af19);
        const lineup=configuredLineup||['heuristic','search','neural',g%2?'guided':'world'];
        const seats=Array.from({length:players},(_,p)=>lineup[(p+Math.floor(g/2))%lineup.length]);
        let state=Sim.create(players,seed,seats.map(t=>P.TYPES[t])),moves=0;const states=[],transitionStart=transitionRows;
        while(!state.gameOver){
            if(moves%config.stride===0)states.push(E.encodeState(state));
            const type=seats[state.currentPlayerId];
            const strategy=type==='guided'?config.strategy:null;
            const action=rng.next()<config.exploration?rng.pick(Sim.candidates(state,{doubleRail:!!strategy})):P.plan(state,{type,world,policy,strategy,depth:config.depth,width:8}).selected;
            const next=Sim.step(state,action).state;
            if(transitionFd!==null){
                const values=Float32Array.from([...E.encodeState(state),...E.encodeAction(action,state,{version:config.actionEncoding}),...E.encodeState(next),action.score/100]);
                fs.writeSync(transitionFd,Buffer.from(values.buffer));transitionRows++;
                counts[action.action]=(counts[action.action]||0)+1;
                if(action.target?.connectionIds)counts.doubleRail=(counts.doubleRail||0)+1;
            }
            state=next;if(++moves>1000)throw Error('Non-terminating value replay game');
        }
        states.push(E.encodeState(state));const scores=Array.from({length:4},(_,p)=>state.players[p]?.vp||0),start=rows;
        for(const s of states){const values=Float32Array.from([...s,...scores]);fs.writeSync(fd,Buffer.from(values.buffer));rows++;}
        games.push({game:g,seed,players,split:g%10===8?'validation':g%10===9?'test':'train',start,end:rows,seats,scores,moves});
        if(transitionFd!==null)transitionGames.push({game:g,seed,split:games.at(-1).split,start:transitionStart,end:transitionRows,seats,
            finalScores:Array.from({length:4},(_,p)=>state.players[p]?({id:state.players[p].id,vp:state.players[p].vp,income:state.players[p].income,money:state.players[p].money}):({id:p,vp:0,income:0,money:0}))});
        parentPort.postMessage({type:'progress'});
    }}finally{fs.closeSync(fd);if(transitionFd!==null)fs.closeSync(transitionFd);}
    return {games,rows,file:config.file,transitionGames,transitionRows,counts,transitionFile:config.transitionFile};
}

async function collect({games=800,players=4,lineup=null,strategy=null,seed=180000001,models='world_model/experiments/score-v1/models',out='world_model/data-score-league',workers=4,depth=2,stride=4,exploration=.1,transitions=null,actionEncoding='legacy'}={}){
    if(!Number.isInteger(games)||games<10||!Number.isInteger(players)||players<2||players>4||!Number.isInteger(workers)||workers<1||workers>16||!Number.isInteger(stride)||stride<1||!Number.isInteger(seed)||!Number.isFinite(exploration)||exploration<0||exploration>1)throw Error('Invalid collection configuration');
    if(typeof lineup==='string')lineup=lineup.split(',').map(x=>x.trim()).filter(Boolean);
    if(lineup&&(!Array.isArray(lineup)||lineup.length<players||lineup.some(t=>!['heuristic','search','neural','world','guided'].includes(t))))throw Error('Invalid replay lineup');
    if(strategy&&!['human-guide-v1','human-chain-v2','human-card-v2'].includes(strategy))throw Error('Invalid teacher strategy');
    if(!['legacy','resource-network-v2'].includes(actionEncoding))throw Error('Invalid action encoding');
    if(strategy&&E.schema.rulesVersion!=='economy-v2')throw Error('Teacher strategy requires economy-v2');
    const sourceFiles=['collect_value.js','planner.js','human_strategy.js','human_intent.js','production_chain.js','simulator.js','encoding.js','inference.js','value_features.js',
        '../js/gameState.js','../js/gameLogic.js','../js/gameData.js','../scripts/autorun.js'];
    const sha=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
    const sourceSHA256=Object.fromEntries(sourceFiles.map(f=>[f,sha(path.join(__dirname,f))]));
    const artifacts=Object.fromEntries(['world-model.json','neural-policy.json'].map(name=>[name,sha(path.join(models,name))]));
    if(fs.existsSync(out)&&fs.readdirSync(out).length)throw Error('Choose an empty replay directory');
    if(transitions){
        if(path.resolve(transitions)===path.resolve(out))throw Error('Transition and value directories must differ');
        if(fs.existsSync(transitions)&&fs.readdirSync(transitions).length)throw Error('Choose an empty transition directory');
        fs.mkdirSync(transitions,{recursive:true});
    }
    fs.mkdirSync(out,{recursive:true});const threads=[],pending=[];let completed=0;
    try{for(let i=0;i<workers;i++){
        const config={start:Math.floor(games*i/workers),end:Math.floor(games*(i+1)/workers),players,lineup,strategy,seed,models:path.resolve(models),file:path.resolve(out,`part-${i}.f32`),depth,stride,exploration,
            transitionFile:transitions?path.resolve(transitions,`part-${i}.f32`):null,actionEncoding};
        const worker=new Worker(__filename,{workerData:config});threads.push(worker);
        pending.push(new Promise((resolve,reject)=>{let done=false;
            worker.on('message',m=>{if(m.type==='progress'){completed++;if(completed%10===0||completed===games)console.log(`Collected ${completed}/${games} complete games`);}
                if(m.type==='result'){done=true;resolve(m.result);}});
            worker.on('error',reject);worker.on('exit',code=>{if(!done)reject(Error(`Replay worker exited without result (${code})`));});
        }));
    }
    const parts=await Promise.all(pending);
    if(sourceFiles.some(f=>sourceSHA256[f]!==sha(path.join(__dirname,f)))||Object.entries(artifacts).some(([f,h])=>h!==sha(path.join(models,f))))throw Error('Teacher source or model changed during collection');
    const fd=fs.openSync(path.join(out,'states.f32'),'wx');let rows=0;const records=[];
    try{for(const part of parts){fs.writeSync(fd,fs.readFileSync(part.file));records.push(...part.games.map(g=>({...g,start:g.start+rows,end:g.end+rows})));rows+=part.rows;}}
    finally{fs.closeSync(fd);}
    const config={games,players,lineup,strategy,seed,models,workers,depth,stride,exploration,actionEncoding};
    const provenance={source:'synthetic-teacher-selfplay',humanDemonstrations:0,sourceSHA256};
    fs.writeFileSync(path.join(out,'schema.json'),JSON.stringify({...E.schema,stateDim:E.fields.length,rowWidth:E.fields.length+4,rows,config,artifacts,provenance,format:'float32 [observable state, four final VP labels; unused player labels are zero]'}));
    fs.writeFileSync(path.join(out,'games.json'),JSON.stringify(records));
    if(transitions){
        const sample=Sim.create(4,seed),actionDim=E.encodeAction(Sim.candidates(sample)[0],sample).length;
        const transitionFd=fs.openSync(path.join(transitions,'transitions.f32'),'wx'),transitionGames=[],counts={},splits={train:[],validation:[],test:[]};let total=0;
        try{for(const part of parts){
            fs.writeSync(transitionFd,fs.readFileSync(part.transitionFile));
            for(const g of part.transitionGames){const record={...g,start:g.start+total,end:g.end+total};transitionGames.push(record);
                for(let row=record.start;row<record.end;row++)splits[g.split].push(row);}
            total+=part.transitionRows;for(const [kind,n] of Object.entries(part.counts))counts[kind]=(counts[kind]||0)+n;
        }}finally{fs.closeSync(transitionFd);}
        fs.writeFileSync(path.join(transitions,'schema.json'),JSON.stringify({...E.schema,stateDim:E.fields.length,actionDim,
            rowWidth:2*E.fields.length+actionDim+1,rows:total,games,players,seed,config,artifacts,provenance,counts,actionEncoding,
            format:'little-endian float32 [state,action,next_state,heuristic_score/100]'}));
        fs.writeFileSync(path.join(transitions,'games.json'),JSON.stringify(transitionGames));
        fs.writeFileSync(path.join(transitions,'splits.json'),JSON.stringify(splits));
        for(const part of parts)fs.unlinkSync(part.transitionFile);
    }
    for(const part of parts)fs.unlinkSync(part.file);
    console.log(JSON.stringify({games,players,rows,out,averageVP:records.reduce((s,g)=>s+g.scores.slice(0,players).reduce((a,b)=>a+b,0),0)/(games*players)}));
    }finally{await Promise.all(threads.map(w=>w.terminate()));}
}
if(!isMainThread){try{parentPort.postMessage({type:'result',result:shard(workerData)});}catch(e){throw e;}}
module.exports={collect};
