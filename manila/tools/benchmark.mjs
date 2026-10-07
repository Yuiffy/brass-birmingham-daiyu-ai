import { mkdir, writeFile } from 'node:fs/promises';
import { createGame, observation } from '../engine.mjs';
import { analyzeAuction, simulateGame } from '../ai.mjs';
const args=process.argv.slice(2), option=(key,fallback)=>args.includes(key)?args[args.indexOf(key)+1]:fallback;
const samples=Number(option('--samples',64)),maxBid=Number(option('--max-bid',40)),horizon=Number(option('--horizon',0)),seed=Number(option('--seed',20261007));
const count=Number(option('--players',4)),styles=option('--styles','balanced,cautious,aggressive').split(','),game=createGame(count,seed),results=[];
for(const style of styles){
  const started=Date.now();
  const result=await analyzeAuction(game,0,{samples,maxBid,horizon,style,seed});
  results.push(result);console.log(JSON.stringify({style,samples,horizon,breakEven:result.breakEven,conservativeMax:result.conservativeMax,rangeCensored:result.rangeCensored,elapsedSeconds:(Date.now()-started)/1000}));
}
const report={createdAt:new Date().toISOString(),publicPosition:observation(game,0),results};
await mkdir('output',{recursive:true});await writeFile('output/bid-benchmark.json',JSON.stringify(report,null,2));
console.log('Saved output/bid-benchmark.json');
