import { Worker } from 'node:worker_threads';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { openSync, writeSync, closeSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { availableParallelism } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { strategyDefinition } from '../strategies.mjs';
import { aggregateMatches, pairedVsBalanced } from '../tournament.mjs';

export async function runTournament({tasks,config,output,workers=4}) {
  if(!tasks.length)throw Error('赛程不能为空');
  const root=path.dirname(path.dirname(fileURLToPath(import.meta.url)));
  workers=Math.max(1,Math.min(workers,availableParallelism(),tasks.length));
  await mkdir(output,{recursive:true});
  const hashes={};
  for(const file of ['engine.mjs','ai.mjs','strategies.mjs','tournament.mjs','league.mjs','tools/tournament-runner.mjs','tools/tournament-worker.mjs','tools/evolve.mjs'])hashes[file]=createHash('sha256').update(await readFile(path.join(root,file))).digest('hex');
  const ids=[...new Set(tasks.flatMap(t=>t.lineup))];
  const manifest={schema:2,config,workers,scheduledGames:tasks.length,node:process.version,sourceSHA256:hashes,startedAt:new Date().toISOString(),strategyDescriptions:Object.fromEntries(ids.map(id=>[id,strategyDefinition(id)]))};
  await writeFile(path.join(output,'manifest.json'),JSON.stringify(manifest,null,2));
  const raw=openSync(path.join(output,'matches.jsonl'),'w'),matches=[],pool=[];
  let next=0,complete=0,lastProgress=performance.now(),failed=false;
  const start=performance.now();
  console.log(JSON.stringify({event:'start',stage:config.stage,games:tasks.length,workers,output}));
  try{
    await new Promise((resolve,reject)=>{
      const fail=e=>{if(failed)return;failed=true;for(const w of pool)w.terminate();reject(e);};
      const dispatch=w=>{if(next<tasks.length)w.postMessage({...tasks[next],id:next++});};
      for(let i=0;i<workers;i++){
        const worker=new Worker(new URL('./tournament-worker.mjs',import.meta.url),{workerData:{validate:config.validate,searchOptions:config.searchOptions}});pool.push(worker);
        worker.on('error',fail);
        worker.on('exit',code=>{if(complete<tasks.length&&!failed)fail(Error(`Worker exited early: ${code}`));});
        worker.on('message',({id,match,error})=>{
          if(failed)return;
          if(error){fail(Error(`Match ${id}: ${error}`));return;}
          try{
            const result={...match,id};matches.push(result);writeSync(raw,JSON.stringify(result)+'\n');complete++;
            const now=performance.now();
            if(now-lastProgress>10000||complete===tasks.length){lastProgress=now;console.log(JSON.stringify({event:'progress',stage:config.stage,complete,total:tasks.length,elapsedSeconds:(now-start)/1000}));}
            if(complete===tasks.length){resolve();return;}
            dispatch(worker);
          }catch(e){fail(e);}
        });dispatch(worker);
      }
    });
  }finally{closeSync(raw);await Promise.allSettled(pool.map(w=>w.terminate()));}
  matches.sort((a,b)=>a.id-b.id);
  const summary=aggregateMatches(matches,config.mode),report={...manifest,finishedAt:new Date().toISOString(),completedGames:complete,elapsedSeconds:(performance.now()-start)/1000,summary,pairedVsBalanced:pairedVsBalanced(summary)};
  await writeFile(path.join(output,'summary.json'),JSON.stringify(report,null,2));
  const csv=['players,strategy,n,seed_blocks,mean,min,max,sd,p05,p25,median,p75,p95,mean95_low,mean95_high,win_rate,win95_low,win95_high,average_rank,average_voyages,average_captain_wins,average_bid_spent,average_mortgages',...summary.map(r=>[r.players,r.strategy,r.n,r.seedBlocks,r.mean,r.min,r.max,r.sd,r.p05,r.p25,r.median,r.p75,r.p95,...(r.mean95||[null,null]),r.winRate,...(r.win95||[null,null]),r.averageRank,r.averageVoyages,r.averageCaptainWins,r.averageBidSpent,r.averageMortgages].join(','))].join('\n')+'\n';
  await writeFile(path.join(output,'summary.csv'),csv);
  console.log(JSON.stringify({event:'done',stage:config.stage,complete,elapsedSeconds:report.elapsedSeconds,summary:summary.map(({blocks,histogram,bySeat,...r})=>r)}));
  return {report,matches};
}
