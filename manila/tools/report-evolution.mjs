import { readFile, writeFile, mkdir, readdir } from 'node:fs/promises';
import { gzipSync } from 'node:zlib';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe } from '../tournament.mjs';
import { strategyDefinition } from '../strategies.mjs';

const root=path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const input=path.resolve(process.argv[2]||'output/evolution'),output=path.resolve(process.argv[3]||'reports/2026-10-07-evolution');
const sha=data=>createHash('sha256').update(data).digest('hex');
const directories=(await readdir(input,{withFileTypes:true})).filter(d=>d.isDirectory()&&d.name!=='specs').map(d=>d.name).sort();
await mkdir(output,{recursive:true});
await mkdir(path.join(output,'specs'),{recursive:true});
const stages=[],artifactSHA256={},seedSet=new Set(),strategySet=new Set();
const compact=r=>{const {blocks,...rest}=r;return rest;};
const paired=(rows,baseline)=>rows.filter(r=>r.strategy!==baseline).map(r=>{
  const base=rows.find(b=>b.strategy===baseline),bySeed=new Map(base.blocks.map(b=>[b.seed,b]));
  const stats=key=>{const values=r.blocks.map(b=>b[key]-bySeed.get(b.seed)[key]),d=describe(values);return {mean:d.mean,ci95:d.n>1?[d.mean-1.96*d.sd/Math.sqrt(d.n),d.mean+1.96*d.sd/Math.sqrt(d.n)]:null,n:d.n};};
  return {challenger:r.strategy,baseline,win:stats('win'),score:stats('mean')};
});
for(const directory of directories){
  const selection=JSON.parse(await readFile(path.join(input,directory,'selection.json'),'utf8'));
  const run=JSON.parse(await readFile(path.join(input,directory,'summary.json'),'utf8'));
  const raw=await readFile(path.join(input,directory,'matches.jsonl'),'utf8'),matches=raw.trim().split('\n').map(JSON.parse).sort((a,b)=>a.id-b.id);
  for(const m of matches){seedSet.add(m.seed);for(const id of m.lineup)strategySet.add(id);}
  if(matches.length!==run.scheduledGames||matches.length!==run.completedGames||matches.some((m,i)=>m.id!==i||Math.max(...m.market)!==5||Math.abs(m.results.reduce((v,r)=>v+r.win,0)-1)>1e-9||m.results.some(r=>r.score!==r.cash+r.stockValue-15*r.mortgages)))throw Error(`Invalid results: ${directory}`);
  for(const [file,hash] of Object.entries(run.sourceSHA256)){
    const bytes=await readFile(path.join(root,file));
    if(sha(bytes)!==hash&&sha(bytes.toString('utf8').replaceAll('\r\n','\n'))!==run.sourceLF_SHA256?.[file])throw Error(`Source changed: ${file}`);
  }
  const filename=directory+'.jsonl.gz',compressed=gzipSync(matches.map(m=>JSON.stringify(m)).join('\n')+'\n',{level:9});
  await writeFile(path.join(output,filename),compressed);artifactSHA256[filename]=sha(compressed);
  const auction=selection.summary.map(r=>{
    const events=matches.flatMap(m=>m.auctionHistory.filter(h=>m.lineup[h.identity]===r.strategy).map(h=>({...h,seed:m.seed})));
    const entries=matches.flatMap(m=>m.results.filter(p=>p.strategy===r.strategy));
    return {strategy:r.strategy,prices:describe(events.map(h=>h.price)),byMarket:Array.from({length:5},(_,level)=>({level,...describe(events.filter(h=>Math.max(...h.marketBefore)===level).map(h=>h.price))})),byVoyage:Array.from(new Set(events.map(h=>h.voyage))).sort((a,b)=>a-b).map(voyage=>({voyage,...describe(events.filter(h=>h.voyage===voyage).map(h=>h.price))})),negativeScoreRate:entries.filter(p=>p.score<0).length/entries.length,zeroCashRate:entries.filter(p=>p.cash===0).length/entries.length};
  });
  const {summary:unused,pairedVsBalanced:unused2,...manifest}=run;
  const c=run.config,format=c.finalists?(c.format==='mixed'?'mixed':'final'):c.challengers?'challenge':'pair';
  const spec={stage:directory,format,seeds:c.seeds,base:c.base,searchOptions:c.searchOptions,...(format==='pair'?{strategies:c.strategies}:format==='challenge'?{challengers:c.challengers,opponents:c.opponents||Array(3).fill(c.resident),resident:c.resident}:{finalists:c.finalists}),selectionReason:c.selectionReason||c.selectedFrom||'Initial preset league stage'};
  const specText=JSON.stringify(spec,null,2)+'\n';await writeFile(path.join(output,'specs',directory+'.json'),specText);artifactSHA256['specs/'+directory+'.json']=sha(specText);
  stages.push({stage:directory,manifest,summary:selection.summary.map(compact),challenge:selection.challenge?.map(compact),pairDetails:selection.pairDetails.map(p=>({...p,summary:p.summary.map(compact)})),paired:selection.paired,auditPaired:selection.config.resident?paired(selection.challenge,selection.config.resident):undefined,auction});
}
const sourceLF_SHA256={};for(const file of Object.keys(stages[0].manifest.sourceSHA256))sourceLF_SHA256[file]=sha((await readFile(path.join(root,file),'utf8')).replaceAll('\r\n','\n'));
const summary={schema:1,generatedAt:new Date().toISOString(),completedGames:stages.reduce((v,s)=>v+s.manifest.completedGames,0),uniqueSeeds:seedSet.size,strategies:[...strategySet].sort(),failedGames:0,truncatedGames:0,sourceLF_SHA256,artifactSHA256,stages};
await writeFile(path.join(output,'summary.json'),JSON.stringify(summary,null,2)+'\n');
const columns=['stage','strategy','n','seedBlocks','mean','min','max','sd','p05','p25','median','p75','p95','winRate','averageRank','averageCaptainWins','averageBidSpent','averageMortgages'];
await writeFile(path.join(output,'summary.csv'),[columns.join(','),...stages.flatMap(s=>(s.challenge||s.summary).map(r=>columns.map(k=>k==='stage'?s.stage:r[k]).join(',')))].join('\n')+'\n');
const f=v=>v.toFixed(2),pct=v=>(100*v).toFixed(1)+'%',ci=(v,scale=1)=>v?v.map(x=>f(x*scale)).join(' ～ '):'样本不足';
const table=rows=>['| 策略 | 观察数 | 种子数 | 胜率 [95%区间] | 平均财富 | 最低 | 最高 | P05 | 中位数 | P95 | 平均抵押股数 |','| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',...rows.map(r=>`| ${r.name} | ${r.n} | ${r.seedBlocks} | ${pct(r.winRate)} [${ci(r.win95,100)}]% | ${f(r.mean)} | ${r.min} | ${r.max} | ${f(r.p05)} | ${f(r.median)} | ${f(r.p95)} | ${f(r.averageMortgages)} |`)].join('\n');
const title=summary.strategies.some(id=>id.startsWith('formula-'))?'竞拍估值公式替换实验明细':'竞价升级与强策略联赛明细';
const text=['# '+title,'',`完整终局 ${summary.completedGames} 局；失败 / 非法动作 / 截断均为 0。分数 = 现金 + 股票市值 − 15 × 抵押股票数。`, '',...stages.flatMap(s=>[`## ${s.stage}`,'','配置：`'+JSON.stringify(s.manifest.config)+'`。','',s.challenge?'以下只统计身份 0 的挑战者；陪测对手不混入。':'以下统计全部参赛身份，各候选在赛程中的出现次数平衡。','',table(s.challenge||s.summary),'',...(s.auditPaired?['| 挑战者 | 相对居民自我对战的胜率差（百分点） | 95%区间 | 财富差 | 95%区间 |','| --- | ---: | --- | ---: | --- |',...s.auditPaired.map(p=>`| ${strategyDefinition(p.challenger).name} | ${f(p.win.mean*100)} | ${ci(p.win.ci95,100)} | ${f(p.score.mean)} | ${ci(p.score.ci95)} |`),'']:[]),...(s.paired?['| 同桌配对 | 胜率差（百分点） | 95%区间 | 财富差 | 95%区间 |','| --- | ---: | --- | ---: | --- |',...s.paired.map(p=>`| ${p.a} − ${p.b} | ${f(p.win.mean*100)} | ${ci(p.win.ci95,100)} | ${f(p.score.mean)} | ${ci(p.score.ci95)} |`),'']:[])]),'## 成交价与尾部风险','','此表统计该阶段所有该策略身份赢得船长时的价格；挑战者阶段也包含陪测同策略身份，不能与挑战者专属胜率直接视为同一子样本。','', '| 阶段 | 策略 | 船长事件数 | 成交均价 | 中位数 | P05–P95 | 最低–最高 | 负分比例 |','| --- | --- | ---: | ---: | ---: | --- | --- | ---: |',...stages.flatMap(s=>s.auction.map(a=>`| ${s.stage} | ${strategyDefinition(a.strategy).name} | ${a.prices.n} | ${a.prices.n?f(a.prices.mean):'—'} | ${a.prices.n?f(a.prices.median):'—'} | ${a.prices.n?ci([a.prices.p05,a.prices.p95]):'—'} | ${a.prices.n?ci([a.prices.min,a.prices.max]):'—'} | ${pct(a.negativeScoreRate)} |`)),'','均值和胜率区间按种子组计算，同种子的座位 / 组合不当作独立样本。筛选阶段用于选择候选，决赛与居民测试采用不同的新种子。区间未做多重比较校正；不把筛选胜者的最高观测均值当成已证明的全局最优。逐对入侵矩阵、逐市场阶段成交价与源哈希见 summary.json，逐局日志在各阶段 jsonl.gz。',''];
await writeFile(path.join(output,'RESULTS.md'),text.join('\n'));
console.log(JSON.stringify({output,games:summary.completedGames,stages:stages.map(s=>({stage:s.stage,config:s.manifest.config,ranking:(s.challenge||s.summary).map(r=>({id:r.strategy,win:r.winRate,mean:r.mean})),auditPaired:s.auditPaired}))}));
