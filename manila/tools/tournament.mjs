import { Worker } from 'node:worker_threads';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { createWriteStream } from 'node:fs';
import { createHash } from 'node:crypto';
import { availableParallelism } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { STRATEGIES } from '../strategies.mjs';
import { buildSchedule, aggregateMatches, pairedVsBalanced } from '../tournament.mjs';

const args=process.argv.slice(2),option=(key,fallback)=>args.includes(key)?args[args.indexOf(key)+1]:fallback;
const config={mode:option('--mode','focal'),seeds:Number(option('--seeds',128)),startSeed:Number(option('--seed',20261007)),counts:option('--players','3,4,5').split(',').map(Number),strategies:option('--strategies','balanced,cautious,aggressive,greedy,random,search').split(','),searchOptions:{samples:Number(option('--search-samples',8)),width:Number(option('--search-width',4))},validate:args.includes('--validate')};
if (!Object.values(config.searchOptions).every(n=>Number.isInteger(n)&&n>0)) throw Error('搜索样本数与候选宽度必须是正整数');
if (new Set(config.strategies).size!==config.strategies.length || new Set(config.counts).size!==config.counts.length) throw Error('策略与人数列表不能重复');
if (config.mode==='mixed' && config.counts.some(n=>n>config.strategies.length)) throw Error('混合赛需要至少与人数一样多的不同策略');
const schedule=buildSchedule(config),root=path.dirname(path.dirname(fileURLToPath(import.meta.url)));
if (!schedule.length) throw Error('赛程不能为空');
const output=path.resolve(option('--output',`output/tournament-${config.mode}`));
const workers=Math.max(1,Math.min(Number(option('--workers',4)),availableParallelism(),schedule.length));
await mkdir(output,{recursive:true});
const hashes={};for(const file of ['engine.mjs','ai.mjs','strategies.mjs','tournament.mjs'])hashes[file]=createHash('sha256').update(await readFile(path.join(root,file))).digest('hex');
const manifest={schema:1,config,workers,scheduledGames:schedule.length,node:process.version,sourceSHA256:hashes,startedAt:new Date().toISOString(),strategyDescriptions:STRATEGIES};
await writeFile(path.join(output,'manifest.json'),JSON.stringify(manifest,null,2));
const raw=createWriteStream(path.join(output,'matches.jsonl'));
const matches=[],pool=[];let next=0,complete=0,lastProgress=0,failed=false;
const start=performance.now();
console.log(JSON.stringify({event:'start',games:schedule.length,workers,config,output}));
await new Promise((resolve,reject)=>{
  const fail=e=>{if(failed)return;failed=true;for(const w of pool)w.terminate();raw.end();reject(e);};
  const dispatch=w=>{if(next<schedule.length)w.postMessage(schedule[next++]);};
  for(let i=0;i<workers;i++){
    const worker=new Worker(new URL('./tournament-worker.mjs',import.meta.url),{workerData:{validate:config.validate,searchOptions:config.searchOptions}});pool.push(worker);
    worker.on('error',fail);
    worker.on('message',({id,match,error})=>{
      if(error){fail(Error(`对局 ${id} 失败：${error}`));return;}
      if(failed)return;
      matches.push({...match,id});raw.write(JSON.stringify({...match,id})+'\n');complete++;
      const now=performance.now();
      if(now-lastProgress>10000||complete===schedule.length){lastProgress=now;console.log(JSON.stringify({event:'progress',complete,total:schedule.length,elapsedSeconds:(now-start)/1000}));}
      if(complete===schedule.length){for(const w of pool)w.terminate();raw.end(resolve);return;}
      dispatch(worker);
    });dispatch(worker);
  }
});
matches.sort((a,b)=>a.id-b.id);
const summary=aggregateMatches(matches,config.mode),report={...manifest,finishedAt:new Date().toISOString(),completedGames:complete,elapsedSeconds:(performance.now()-start)/1000,summary,pairedVsBalanced:pairedVsBalanced(summary)};
await writeFile(path.join(output,'summary.json'),JSON.stringify(report,null,2));
const csv=['players,strategy,n,seed_blocks,mean,min,max,sd,p05,p25,median,p75,p95,mean95_low,mean95_high,win_rate,win95_low,win95_high,average_rank,average_voyages,average_captain_wins,average_bid_spent,average_mortgages',...summary.map(r=>[r.players,r.strategy,r.n,r.seedBlocks,r.mean,r.min,r.max,r.sd,r.p05,r.p25,r.median,r.p75,r.p95,...r.mean95,r.winRate,...r.win95,r.averageRank,r.averageVoyages,r.averageCaptainWins,r.averageBidSpent,r.averageMortgages].join(','))].join('\n')+'\n';
await writeFile(path.join(output,'summary.csv'),csv);
console.log(JSON.stringify({event:'done',complete,elapsedSeconds:report.elapsedSeconds,summary:summary.map(({blocks,histogram,bySeat,...r})=>r)}));
