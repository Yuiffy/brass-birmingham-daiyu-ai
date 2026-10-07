// node tools/report-tournament.mjs [output directory]
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { gzipSync, gunzipSync } from 'node:zlib';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { aggregateMatches, pairedVsBalanced, describe } from '../tournament.mjs';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const destination = path.resolve(process.argv[2] || 'reports/2026-10-07');
const archiveIndex = process.argv.indexOf('--from-archive');
const archive = archiveIndex >= 0 ? path.resolve(process.argv[archiveIndex+1]) : null;
const archiveReport = archive ? JSON.parse(await readFile(path.join(archive,'summary.json'),'utf8')) : null;
const sources = ['tournament-focal', 'tournament-search', 'tournament-mixed'];
const sha256 = data => createHash('sha256').update(data).digest('hex');
const runs = [];
for (const source of sources) {
  const directory = path.join(root, 'output', source);
  const report = archive ? archiveReport.runs.find(r=>r.source===source) : JSON.parse(await readFile(path.join(directory, 'summary.json'), 'utf8'));
  let raw;
  if (archive) {
    const filename=source+'.jsonl.gz',compressed=await readFile(path.join(archive,filename));
    if(sha256(compressed)!==archiveReport.artifactSHA256[filename])throw Error(`Archive hash mismatch: ${source}`);
    raw=gunzipSync(compressed).toString('utf8');
  } else raw=await readFile(path.join(directory, 'matches.jsonl'), 'utf8');
  const matches = raw.trim().split('\n').map(JSON.parse).sort((a,b) => a.id-b.id);
  if (matches.length !== report.scheduledGames || report.completedGames !== matches.length || matches.some((m,i)=>m.id!==i)) throw Error(`Incomplete run: ${source}`);
  if (!report.config.validate) throw Error(`Run not validated: ${source}`);
  for (const m of matches) {
    if (m.results.length !== m.players || Math.max(...m.market) !== 5 || Math.abs(m.results.reduce((v,r)=>v+r.win,0)-1)>1e-9 || m.results.some(r=>r.score!==r.cash+r.stockValue-15*r.mortgages)) throw Error(`Invalid terminal result: ${source}/${m.id}`);
  }
  for (const [file, hash] of Object.entries(report.sourceSHA256)) {
    const bytes=await readFile(path.join(root,file)),current=sha256(bytes);
    const matchesLF=archive&&sha256(bytes.toString('utf8').replaceAll('\r\n','\n'))===archiveReport.sourceLF_SHA256[file];
    if (current !== hash&&!matchesLF) throw Error(`Source changed since run: ${file}`);
  }
  runs.push({ source, report, matches });
}
await mkdir(destination,{recursive:true});
const focal = runs.filter(r=>r.report.config.mode==='focal').flatMap(r=>r.matches);
const mixed = runs.filter(r=>r.report.config.mode==='mixed').flatMap(r=>r.matches);
const focalSummary = aggregateMatches(focal);
const mixedSummary = aggregateMatches(mixed,'mixed');
function diagnostics(matches, mode) {
  const output = [];
  for (const row of aggregateMatches(matches,mode)) {
    const entries=matches.filter(m=>m.players===row.players).flatMap(m=>m.results.filter(r=>r.strategy===row.strategy&&(mode!=='focal'||r.identity===0)).map(r=>({...r,elapsedMs:m.elapsedMs})));
    const prices=matches.filter(m=>m.players===row.players).flatMap(m=>m.auctionHistory.filter(a=>m.lineup[a.identity]===row.strategy&&(mode!=='focal'||a.identity===0)).map(a=>a.price));
    output.push({players:row.players,strategy:row.strategy,negativeScoreRate:entries.filter(r=>r.score<0).length/entries.length,finalMortgageRate:entries.filter(r=>r.mortgages>0).length/entries.length,meanCash:describe(entries.map(r=>r.cash)).mean,meanStockValue:describe(entries.map(r=>r.stockValue)).mean,winningAuctionPrices:describe(prices),meanGameWallMs:describe(entries.map(r=>r.elapsedMs)).mean});
  }
  return output;
}
const report={schema:1,generatedAt:new Date().toISOString(),completedGames:focal.length+mixed.length,failedGames:0,truncatedGames:0,runs:runs.map(({source,report})=>({source,...report,summary:undefined,pairedVsBalanced:undefined})),focal:focalSummary,mixed:mixedSummary,pairedFocal:pairedVsBalanced(focalSummary),pairedMixed:pairedVsBalanced(mixedSummary),diagnostics:{focal:diagnostics(focal,'focal'),mixed:diagnostics(mixed,'mixed')}};
report.sourceLF_SHA256={};report.artifactSHA256={};
for(const file of Object.keys(runs[0].report.sourceSHA256))report.sourceLF_SHA256[file]=sha256((await readFile(path.join(root,file),'utf8')).replaceAll('\r\n','\n'));
for(const run of runs){
  const canonical=run.matches.map(m=>JSON.stringify(m)).join('\n')+'\n';
  const filename=run.source+'.jsonl.gz',compressed=gzipSync(canonical,{level:9,mtime:0});
  report.artifactSHA256[filename]=sha256(compressed);
  await writeFile(path.join(destination,filename),compressed);
}
await writeFile(path.join(destination,'summary.json'),JSON.stringify(report,null,2)+'\n');
const columns=['mode','players','strategy','n','seedBlocks','mean','min','max','sd','p05','p25','median','p75','p95','mean95_low','mean95_high','winRate','win95_low','win95_high','averageRank','averageVoyages','averageCaptainWins','averageBidSpent','averageMortgages'];
const rows=[['focal',focalSummary],['mixed',mixedSummary]].flatMap(([mode,rs])=>rs.map(r=>({...r,mode,mean95_low:r.mean95[0],mean95_high:r.mean95[1],win95_low:r.win95[0],win95_high:r.win95[1]})));
await writeFile(path.join(destination,'summary.csv'),[columns.join(','),...rows.map(r=>columns.map(k=>r[k]).join(','))].join('\n')+'\n');
const f=n=>n.toFixed(2), pct=n=>(100*n).toFixed(1)+'%', interval=v=>v.map(f).join(' ～ ');
const table=rs=>['| 策略 | 局数 | 均值 [95%区间] | 最低 | 最高 | P05 | P25 | 中位数 | P75 | P95 | 标准差 | 胜率 |','| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',...rs.map(r=>`| ${r.name} | ${r.n} | ${f(r.mean)} [${interval(r.mean95)}] | ${r.min} | ${r.max} | ${f(r.p05)} | ${f(r.p25)} | ${f(r.median)} | ${f(r.p75)} | ${f(r.p95)} | ${f(r.sd)} | ${pct(r.winRate)} |`)].join('\n');
const text=['# 对战统计明细','',`完整终局 ${report.completedGames} 局；非法动作 / 失败 / 截断均为 0。分数为现金 + 股票市值 − 15 × 抵押股票数。`, '', '## 单策略挑战均衡对手','', '每桌只有一个被评估身份，其余均为均衡 AI。对同一副初始手牌和同一骰子流，轮换全部座位。均衡自身也按同一身份统计，不混入陪测对手。', '', ...[3,4,5].flatMap(n=>[`### ${n} 人`, '',table(focalSummary.filter(r=>r.players===n)),'']), '## 配对财富差：挑战策略减均衡','', '以共同种子的全部座位轮换均值配对；模型前瞻只有 64 个共同种子，其余 128 个。区间描述均值模拟误差。','', '| 人数 | 策略 | 独立种子组 | 配对均值差 | 95%区间 |', '| ---: | --- | ---: | ---: | --- |', ...report.pairedFocal.map(r=>`| ${r.players} | ${focalSummary.find(s=>s.strategy===r.strategy).name} | ${r.n} | ${f(r.scoreDelta)} | ${interval(r.mean95)} |`),'','## 四人混合对战','', '均衡、保守竞价、积极竞价、即时收益各一位，128 个种子 × 24 种座位 / 手牌分配排列。各策略每局出现一次，以下每个策略有 3,072 条观察，128 个独立种子组。', '',table(mixedSummary),'','## 船长与抵押','', '| 模式 | 人数 | 策略 | 平均船长次数 | 整局竞价支出 | 平均终局抵押数 | 成交价均值 | 成交价中位数 | 成交价 P05–P95 | 终局负分比例 |','| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |',...rows.map(r=>{const d=report.diagnostics[r.mode].find(d=>d.players===r.players&&d.strategy===r.strategy),p=d.winningAuctionPrices;return `| ${r.mode} | ${r.players} | ${r.name} | ${f(r.averageCaptainWins)} | ${f(r.averageBidSpent)} | ${f(r.averageMortgages)} | ${p.n?f(p.mean):'—'} | ${p.n?f(p.median):'—'} | ${p.n?interval([p.p05,p.p95]):'—'} | ${pct(d.negativeScoreRate)} |`;}),'','所有均值 / 胜率区间按种子分组计算近似正态 95% 区间，同组座位和排列不当作独立样本。胜率在并列第一时按人数共享。挑战赛的各策略对手环境不同于混合赛，二者不合并。最低 / 最高为观察极值，尤其搜索策略局数较少，不宜直接比较尾部能力。完整直方图、逐座位统计、种子组均值与源文件 SHA256 见 summary.json。',''];
await writeFile(path.join(destination,'RESULTS.md'),text.join('\n'));
console.log(JSON.stringify({destination,games:report.completedGames,focal:focalSummary.map(({blocks,histogram,bySeat,...r})=>r),paired:report.pairedFocal,diagnostics:report.diagnostics}));
