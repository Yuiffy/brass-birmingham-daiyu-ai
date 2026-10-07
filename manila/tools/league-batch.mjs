import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildSchedule } from '../tournament.mjs';
import { pairLeague, challengeLeague, finalLeague, ranked, pairDetails, challengeDetails, pairedFinals } from '../league.mjs';
import { runTournament } from './tournament-runner.mjs';

export async function runLeague(spec,root='output/evolution',workers=8){
  const {stage,format,seeds,base}=spec;
  let tasks;
  if(format==='pair')tasks=pairLeague(spec.strategies,seeds,base,stage);
  else if(format==='challenge')tasks=challengeLeague(spec.challengers,spec.opponents,seeds,base,stage);
  else if(format==='final')tasks=finalLeague(spec.finalists,seeds,base,stage);
  else if(format==='mixed')tasks=buildSchedule({mode:'mixed',strategies:spec.finalists,counts:[4],seeds,startSeed:base}).map(t=>({...t,stage}));
  else throw Error('Invalid league format');
  const config={...spec,mode:'league',validate:true,searchOptions:spec.searchOptions||{samples:4,width:3}};
  const output=path.resolve(root,stage),{report,matches}=await runTournament({tasks,config,output,workers});
  const selection={stage,config,completedGames:report.completedGames,summary:ranked(report.summary),pairDetails:pairDetails(matches),challenge:format==='challenge'?challengeDetails(matches):undefined,paired:spec.finalists?pairedFinals(matches,spec.finalists):undefined};
  await writeFile(path.join(output,'selection.json'),JSON.stringify(selection,null,2));
  console.log(JSON.stringify({event:'selection',stage,ranking:(selection.challenge||selection.summary).map(r=>({strategy:r.strategy,win:r.winRate,wealth:r.mean}))}));
  return selection;
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  const spec=JSON.parse(await readFile(process.argv[2],'utf8'));
  await runLeague(spec,process.argv[3]||'output/evolution',Number(process.argv[4]||8));
}
