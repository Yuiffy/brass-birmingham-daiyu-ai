import { readFile, writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { strategyDefinition } from '../strategies.mjs';
import { pairLeague, challengeLeague, finalLeague, ranked, pairDetails, challengeDetails, pairedFinals } from '../league.mjs';
import { runTournament } from './tournament-runner.mjs';

const args=process.argv.slice(2),option=(key,fallback)=>args.includes(key)?args[args.indexOf(key)+1]:fallback;
const root=path.resolve(option('--output','output/evolution'));
const workers=Number(option('--workers',8)),stage=option('--stage','all');
const searchOptions={samples:4,width:3};
const read=async name=>JSON.parse(await readFile(path.join(root,name,'selection.json'),'utf8'));
const offsets=[3,6,9,12,18,24,36,48,60];

async function run(name,tasks,metadata) {
  const config={stage:name,mode:'league',validate:true,searchOptions,...metadata};
  const {report,matches}=await runTournament({tasks,config,output:path.join(root,name),workers});
  const selection={stage:name,config,completedGames:report.completedGames,summary:ranked(report.summary),pairDetails:pairDetails(matches),challenge:tasks[0].challenger?challengeDetails(matches):undefined,paired:metadata.finalists?pairedFinals(matches,metadata.finalists):undefined};
  await writeFile(path.join(root,name,'selection.json'),JSON.stringify(selection,null,2));
  console.log(JSON.stringify({event:'selection',stage:name,ranking:(selection.challenge||selection.summary).map(r=>({strategy:r.strategy,win:r.winRate,wealth:r.mean}))}));
  return selection;
}

async function screen(){const strategies=offsets.map(n=>'bid+'+n);return run('01-screen',pairLeague(strategies,32,31000001,'01-screen'),{seeds:32,base:31000001,strategies,format:'all pairs; 1:3 / adjacent 2:2 / alternating 2:2 / 3:1; full rotations'});}
async function refine(){
  const previous=await read('01-screen'),top=previous.summary.slice(0,2).map(r=>r.strategy),center=strategyDefinition(top[0]).offset;
  const nearby=[center-3,center-1,center+1,center>=60?96:center+3].map(n=>'bid+'+Math.max(1,n));
  const strategies=[...new Set([...top,...nearby,...top.map(id=>id.replace('bid+','bid-liquid+'))])];
  return run('02-refine',pairLeague(strategies,24,42000001,'02-refine'),{seeds:24,base:42000001,strategies,selectedFrom:previous.stage,format:'same balanced pair league'});
}
async function models(){
  const previous=await read('02-refine'),opponents=previous.summary.slice(0,3).map(r=>r.strategy);
  const challengers=[...opponents.slice(0,2),...opponents.slice(0,2).flatMap(id=>['look','relative'].map(kind=>id.replace('bid',kind)))];
  return run('03-models',challengeLeague(challengers,opponents,32,53000001,'03-models'),{seeds:32,base:53000001,opponents,challengers,selectedFrom:previous.stage,format:'one challenger versus three selected strong heuristics; full rotations'});
}
async function finals(){
  const previous=await read('03-models'),rows=previous.challenge;
  const finalists=[...rows.filter(r=>strategyDefinition(r.strategy).kind==='bid').slice(0,2),...rows.filter(r=>strategyDefinition(r.strategy).kind!=='bid').slice(0,2)].map(r=>r.strategy);
  return run('04-finals',finalLeague(finalists,64,64000001,'04-finals'),{seeds:64,base:64000001,finalists,selectedFrom:previous.stage,format:'frozen four finalists; eight identity permutations per new seed'});
}
async function audit(){
  const name=option('--name','05-audit'),previous=await read('04-finals'),resident=option('--resident',previous.summary[0].strategy);
  const refine=await read('02-refine'),models=await read('03-models');
  const defaults=[...previous.config.finalists,...refine.summary.slice(0,3).map(r=>r.strategy),...models.challenge.filter(r=>strategyDefinition(r.strategy).kind!=='bid').slice(0,3).map(r=>r.strategy),'bid+24','bid+60'];
  const challengers=[...new Set([resident,...option('--challengers',defaults.join(',')).split(',')])];
  const seeds=Number(option('--seeds',32)),base=Number(option('--base',75000001));
  return run(name,challengeLeague(challengers,Array(3).fill(resident),seeds,base,name),{seeds,base,resident,challengers,format:'one challenger versus three copies of resident; resident self-play control'});
}

await mkdir(root,{recursive:true});
if(stage==='all'){await screen();await refine();await models();await finals();await audit();}
else if(stage==='screen')await screen();
else if(stage==='refine')await refine();
else if(stage==='models')await models();
else if(stage==='finals')await finals();
else if(stage==='audit')await audit();
else throw Error('Unknown evolution stage');
