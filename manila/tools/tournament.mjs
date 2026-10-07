import path from 'node:path';
import { buildSchedule } from '../tournament.mjs';
import { runTournament } from './tournament-runner.mjs';

const args=process.argv.slice(2),option=(key,fallback)=>args.includes(key)?args[args.indexOf(key)+1]:fallback;
const config={mode:option('--mode','focal'),seeds:Number(option('--seeds',128)),startSeed:Number(option('--seed',20261007)),counts:option('--players','3,4,5').split(',').map(Number),strategies:option('--strategies','balanced,cautious,aggressive,greedy,random,search').split(','),searchOptions:{samples:Number(option('--search-samples',8)),width:Number(option('--search-width',4))},validate:args.includes('--validate')};
if (!Object.values(config.searchOptions).every(n=>Number.isInteger(n)&&n>0)) throw Error('搜索样本数与候选宽度必须是正整数');
if (new Set(config.strategies).size!==config.strategies.length || new Set(config.counts).size!==config.counts.length) throw Error('策略与人数列表不能重复');
if (config.mode==='mixed' && config.counts.some(n=>n>config.strategies.length)) throw Error('混合赛需要至少与人数一样多的不同策略');
await runTournament({tasks:buildSchedule(config),config,output:path.resolve(option('--output',`output/tournament-${config.mode}`)),workers:Number(option('--workers',4))});
