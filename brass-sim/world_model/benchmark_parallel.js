// Parallelize whole seed groups. Each worker keeps every seat rotation.
const {Worker,isMainThread,parentPort,workerData}=require('node:worker_threads');
const path=require('node:path'),fs=require('node:fs');
const {tournament}=require('./tournament');
const {Network}=require('./inference');

async function parallelTournament({workers=4,games=12,seed=800001,models='world_model/models',guidedModels=null,depth=2,width=8,types=['heuristic','search','neural','world'],onProgress=()=>{}}={}){
    if(!Array.isArray(types)||types.length<2||types.length>4)throw Error('Use 2–4 seats');
    const seats=types.length;
    if(!Number.isInteger(workers)||workers<1||workers>16||!Number.isInteger(games)||games<seats||games%seats)throw Error('Use 1–16 workers and complete seat rotations');
    const count=Math.min(workers,games/seats),threads=[],results=[];
    let groupStart=0,completed=0;
    try{
        const pending=[];
        for(let i=0;i<count;i++){
            const groups=Math.floor(games/seats/count)+(i<games/seats%count?1:0);
            const offset=groupStart*seats;
            const worker=new Worker(__filename,{workerData:{games:groups*seats,seed:seed+groupStart*9973,models:path.resolve(models),guidedModels:guidedModels?path.resolve(guidedModels):null,depth,width,types}});
            groupStart+=groups;threads.push(worker);
            pending.push(new Promise((resolve,reject)=>{
                let received=false;
                worker.on('message',message=>{
                    if(message.type==='progress'){completed++;onProgress({game:completed,games,completed:true,scores:message.scores});}
                    if(message.type==='result'){received=true;results.push({offset,report:message.report});resolve();}
                });
                worker.on('error',reject);
                worker.on('exit',code=>{if(!received)reject(Error(`Benchmark worker exited without a report (${code})`));});
            }));
        }
        await Promise.all(pending);
    }finally{await Promise.all(threads.map(t=>t.terminate()));}
    results.sort((a,b)=>a.offset-b.offset);
    const report=results[0].report;
    const rows=report.rows.map(r=>{
        const all=results.map(x=>x.report.rows.find(p=>p.type===r.type));
        const sum=key=>all.reduce((n,p)=>n+p[key],0);
        const total={type:r.type,games:sum('games'),wins:sum('wins'),vp:sum('vp'),moves:sum('moves'),thinkingMs:sum('thinkingMs')};
        return {...total,averageVP:total.vp/total.games,winRate:total.wins/total.games,millisecondsPerMove:total.thinkingMs/total.moves};
    });
    return {...report,config:{...report.config,games,seed},execution:{workers:count},completedGames:games,rows,
        games:results.flatMap(({offset,report:r})=>r.games.map(g=>({...g,game:offset+g.game}))),
        notes:report.notes.concat('Whole seed groups run on independent workers; timings include CPU contention.')};
}

if(!isMainThread){
    const load=file=>new Network(JSON.parse(fs.readFileSync(path.join(workerData.models,file),'utf8')));
    tournament({...workerData,world:load('world-model.json'),policy:load('neural-policy.json'),guidedWorld:workerData.guidedModels?new Network(JSON.parse(fs.readFileSync(path.join(workerData.guidedModels,'world-model.json'),'utf8'))):null,
        onProgress:p=>{if(p.completed)parentPort.postMessage({type:'progress',scores:p.scores});}})
        .then(report=>{parentPort.postMessage({type:'result',report});parentPort.close();})
        .catch(error=>{throw error;});
}
module.exports={parallelTournament};
