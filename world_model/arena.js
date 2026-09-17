'use strict';
const names={heuristic:'原启发式 AI',search:'搜索树逻辑型',neural:'神经网络型',world:'世界模型型',guided:'学习增强搜索'};
const $=id=>document.getElementById(id),pct=v=>`${(v*100).toFixed(1)}%`,num=v=>Number(v).toFixed(3);
const vp=v=>Number(v).toLocaleString('zh-CN',{minimumFractionDigits:2,maximumFractionDigits:2});
const escapeHTML=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let worker=null,lastReport=null,versionRequest=0;
const versions={current:{models:'models',reports:'reports',tournament:'tournament.json',strong:'tournament-4p-default-120-new.json'},
    'pre-geography':{models:'experiments/score-v2/models',reports:'experiments/geography-study',tournament:'baseline-world-final.json',guided:'baseline-guided-final.json',evaluation:'experiments/structured-value-10k/reports'},
    'pre-score':{models:'experiments/structured-value-10k/models',reports:'experiments/score-study',tournament:'baseline-world-final.json',guided:'baseline-guided-final.json',evaluation:'experiments/structured-value-10k/reports'},
    v1:{models:'experiments/v1/models',reports:'experiments/v1/reports',tournament:'tournament-400.json'},
    'longer-1k':{models:'experiments/longer-1k/models',reports:'experiments/longer-1k/reports',tournament:'tournament-400.json'},
    'expanded-10k':{models:'experiments/expanded-10k/models',reports:'experiments/expanded-10k/reports',tournament:'tournament-400.json'}};
const read=async url=>{const r=await fetch(url);if(!r.ok)throw Error(`HTTP ${r.status}`);return r.json();};
function syncPlayers(){
    const enabled=$('arena-version').value==='current'&&$('arena-lineup').value==='strong';
    $('arena-players').disabled=!enabled;
    if(!enabled)$('arena-players').value='4';
    const players=Number($('arena-players').value),input=$('arena-games');
    input.min=players;input.step=players;
    const games=Number(input.value);
    if(!Number.isInteger(games)||games<players||games%players)input.value=players*Math.max(1,Math.ceil((games||12)/players));
    return players;
}
function renderTournament(report,saved=false){
    lastReport=report;
    const earned=type=>{const scores=report.games.flatMap(g=>g.scores.filter(p=>p.type===type));return scores.length&&scores.every(p=>Number.isFinite(p.vpWithoutIncomeBonus))?vp(scores.reduce((n,p)=>n+p.vpWithoutIncomeBonus,0)/scores.length):'未记录';};
    $('tournament-result').querySelector('tbody').innerHTML=report.rows.slice().sort((a,b)=>b.averageVP-a.averageVP||b.winRate-a.winRate).map(r=>`<tr><td>${names[r.type]}</td><td>${r.games}</td><td><strong>${vp(r.averageVP)}</strong></td><td>${earned(r.type)}</td><td>${pct(r.winRate)}</td><td>${r.millisecondsPerMove.toFixed(2)}</td></tr>`).join('');
    $('arena-status').textContent=`${saved?'已保存的真实实验':'本次实验'} · ${report.config.types.length} 人 · 完成 ${report.completedGames} 局 · 种子 ${report.config.seed} · 深度 ${report.config.depth}${report.cancelled?' · 已停止':''}${report.balancedSeats?' · 座位轮换完整':' · 座位轮换未完成，请谨慎比较'}`;
    $('arena-export').disabled=false;$('arena-progress').value=report.completedGames/report.config.games*100;
    $('arena-details').innerHTML=`<p class="ai-note">保留分支宽度 ${report.config.width}；新版可先预比较最多 32 个动作，候选方法见导出的模型信息；世界模型使用${report.models?.world?.planningValue==='learned'?'学习到的终局价值':'启发式评分'}。搜索使用启发式评分，学习增强搜索使用终局价值。计算时间未强行等额。停止时未完成的对局不计成绩。</p><div class="comparison-scroll"><table class="ai-table"><thead><tr><th>局 / 种子</th><th>按座位列出 AI 与 VP</th></tr></thead><tbody>${report.games.map(g=>`<tr><td>${g.game} / ${g.seed}</td><td>${g.scores.map(p=>`${names[p.type]} ${p.vp}`).join(' · ')}</td></tr>`).join('')}</tbody></table></div>`;
}
function renderEvaluation(r){
    $('eval-caption').textContent=`评估数据池 ${r.datasetGames.toLocaleString()} 局 · 测试 ${r.testGames} 局 / ${r.testTransitions.toLocaleString()} 条 transition · 所选权重累计第 ${r.modelEpoch} 轮`;
    $('eval-summary').innerHTML=[[pct(r.currentPlayerAccuracy),'下一行动玩家准确率'],[num(r.groups.money.mae),'金钱 MAE / 英镑'],[pct(r.overall.changeRecall),'真实变化字段召回率'],[pct(r.overall.changeF1),'变化检测 F1']].map(([value,label])=>`<div class="metric-tile"><strong>${value}</strong><small>${label}</small></div>`).join('');
    $('eval-fields').innerHTML=`<table class="ai-table"><thead><tr><th>字段组</th><th>原单位 MAE</th><th>取整准确率</th><th>变化字段 MAE</th></tr></thead><tbody>${Object.entries(r.groups).map(([key,m])=>`<tr><td>${key}</td><td>${num(m.mae)}</td><td>${pct(m.roundedAccuracy)}</td><td>${m.changedMAE===null?'无样本':num(m.changedMAE)}</td></tr>`).join('')}</tbody></table>`;
    const max=Math.max(...r.rollout.map(x=>x.normalizedMAE),.001);
    $('eval-rollout').innerHTML=`<div class="rollout-bars">${r.rollout.map(x=>`<div><span>${x.depth} 步</span><i style="width:${x.normalizedMAE/max*65}%"></i><b>${num(x.normalizedMAE)}</b></div>`).join('')}</div><table class="ai-table"><thead><tr><th>步数</th><th>模型 MAE</th><th>状态不变 MAE</th></tr></thead><tbody>${r.rollout.map(x=>`<tr><td>${x.depth}</td><td>${num(x.normalizedMAE)}</td><td>${num(x.persistenceMAE)}</td></tr>`).join('')}</tbody></table>`;
    $('eval-caveat').textContent=`当前模型整体误差${r.rollout[0].normalizedMAE>r.rollout[0].persistenceMAE?'高于':'低于'}“状态不变”基线。多数格子每步都不变，不能只看总体准确率；需结合变化字段召回率、精确率 ${pct(r.overall.changePrecision)} 和实际对战。`;
    $('eval-actions').innerHTML=`<table class="ai-table"><thead><tr><th>动作</th><th>测试样本</th><th>标准化 MAE</th><th>变化字段 MAE</th></tr></thead><tbody>${Object.entries(r.byAction).map(([action,m])=>`<tr><td>${action}</td><td>${m.count}</td><td>${num(m.normalizedMAE)}</td><td>${num(m.changedFieldMAE)}</td></tr>`).join('')}</tbody></table>`;
    $('eval-samples').innerHTML=r.samples.map(s=>`<details><summary>Transition #${s.row}</summary><div class="comparison-scroll"><table class="ai-table"><thead><tr><th>字段</th><th>当前</th><th>预测</th><th>真实</th></tr></thead><tbody>${s.fields.map(f=>`<tr><td>${escapeHTML(f.field)}</td><td>${num(f.before)}</td><td>${num(f.predicted)}</td><td>${num(f.actual)} ${f.correct?'✓':'✗'}</td></tr>`).join('')}</tbody></table></div></details>`).join('');
}
$('arena-start').onclick=()=>{
    const players=Number($('arena-players').value),games=Number($('arena-games').value),seed=Number($('arena-seed').value),depth=Number($('arena-depth').value);
    if(![2,3,4].includes(players)||!Number.isInteger(games)||games<players||games>400||games%players||!Number.isInteger(seed)){$('arena-status').textContent=`请选择 ${players}–400 局，局数为 ${players} 的倍数，并输入整数种子。`;return;}
    worker?.terminate();worker=new Worker('world_model/arena-worker.js');
    versionRequest++;$('arena-version').disabled=true;$('arena-lineup').disabled=true;$('arena-players').disabled=true;
    $('arena-start').disabled=true;$('arena-stop').disabled=false;$('arena-progress').value=0;
    $('arena-status').textContent='正在加载权重并开始对局…';
    const finish=()=>{$('arena-version').disabled=false;$('arena-lineup').disabled=false;$('arena-start').disabled=false;$('arena-stop').disabled=true;worker?.terminate();worker=null;syncPlayers();};
    worker.onmessage=({data})=>{
        if(data.type==='progress'){const p=data.progress;$('arena-status').textContent=`第 ${p.game}/${p.games} 局 · ${p.completed?'已完成':`${p.era} 第 ${p.round} 轮 · ${p.moves} 动作`}`;$('arena-progress').value=(p.game-(p.completed?0:1))/p.games*100;}
        if(data.type==='result'){renderTournament(data.result);finish();}
        if(data.type==='error'){$('arena-status').textContent=`实验失败：${data.message}`;finish();}
    };
    worker.onerror=e=>{$('arena-status').textContent=`实验失败：${e.message}`;finish();};
    const lineup=$('arena-lineup').value;
    const types=lineup==='strong'?Array.from({length:players},(_,i)=>i%2?'world':'guided'):lineup==='guided'?['heuristic','search','neural','guided']:['heuristic','search','neural','world'];
    worker.postMessage({type:'run',config:{games,seed,depth,width:8,types,models:versions[$('arena-version').value].models,currentModels:$('arena-version').value==='current'}});
};
$('arena-stop').onclick=()=>{worker?.postMessage({type:'cancel'});$('arena-stop').disabled=true;$('arena-status').textContent='正在停止，保留已完成对局…';};
$('arena-export').onclick=()=>{if(!lastReport)return;const a=document.createElement('a'),url=URL.createObjectURL(new Blob([JSON.stringify(lastReport,null,2)],{type:'application/json'}));a.href=url;a.download='brass-ai-comparison.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
async function loadVersion(){
    const supportsGuided=['current','pre-score','pre-geography'].includes($('arena-version').value);
    $('arena-lineup').querySelector('option[value="guided"]').disabled=!supportsGuided;
    $('arena-lineup').querySelector('option[value="strong"]').disabled=!supportsGuided;
    if(!supportsGuided)$('arena-lineup').value='standard';
    const players=syncPlayers();
    const version=versions[$('arena-version').value],token=++versionRequest;
    const lineup=$('arena-lineup').value;
    const tournamentFile=lineup==='strong'?version.strong:lineup==='guided'?(version.guided||'tournament-guided.json'):version.tournament;
    const specialized=$('arena-version').value==='current'&&lineup==='strong'&&players<4;
    const tournamentURL=specialized?`world_model/experiments/dynamics-small-study/tournament-${players}p-candidate.json`:tournamentFile?`world_model/${version.reports}/${tournamentFile}`:null;
    const evaluationURL=specialized?`world_model/experiments/dynamics-small-study/prediction-${players}p-candidate/evaluation.json`:`world_model/${version.evaluation||version.reports}/evaluation.json`;
    lastReport=null;$('arena-export').disabled=true;$('tournament-result').querySelector('tbody').innerHTML='';
    $('arena-status').textContent='正在读取所选版本…';
    $('eval-caption').textContent='正在读取所选版本的测试评估…';
    for(const id of ['eval-summary','eval-fields','eval-rollout','eval-caveat','eval-actions','eval-samples','arena-details'])$(id).innerHTML='';
    await Promise.all([
        (tournamentURL?read(tournamentURL):Promise.reject(Error('No saved lineup'))).then(r=>{if(token===versionRequest)renderTournament(r,true);}).catch(()=>{if(token===versionRequest)$('arena-status').textContent='所选版本与阵容还没有保存的对战结果。';}),
        read(evaluationURL).then(r=>{if(token===versionRequest)renderEvaluation(r);}).catch(e=>{if(token===versionRequest)$('eval-caption').textContent=`所选版本评估报告不可用：${e.message}`;})
    ]);
}
$('arena-version').onchange=loadVersion;$('arena-lineup').onchange=loadVersion;$('arena-players').onchange=loadVersion;loadVersion();
read('world_model/reports/higher-scores.json').then(r=>{
    const ci=x=>`${x.estimate>=0?'+':''}${vp(x.estimate)} [${vp(x.low)}, ${vp(x.high)}]`;
    $('growth-caption').textContent=`每种 AI 每版 ${r.runs[0].games} 局 · 全新种子 ${r.protocol.finalSeed} · 固定三个对手 · 4 个座位轮换 · 开发赛选定模型后冻结复测`;
    $('growth-table').innerHTML=`<table class="ai-table"><thead><tr><th>AI</th><th>项目 VP · 前 → 后</th><th>扣除收入加分 · 前 → 后</th><th>扣除后增分 / 95% 区间</th><th>新版项目 VP ≥150</th><th>新版最高项目 VP</th></tr></thead><tbody>${r.runs.map(x=>`<tr><td>${names[x.type]}</td><td>${vp(x.baselineScores.mean)} → <strong>${vp(x.candidateScores.mean)}</strong></td><td>${vp(x.baselineScores.withoutIncomeBonus)} → <strong>${vp(x.candidateScores.withoutIncomeBonus)}</strong></td><td>${ci(x.withoutIncomeBonusGain)}</td><td>${pct(x.candidateScores.atLeast150)}</td><td>${x.candidateScores.maximum}</td></tr>`).join('')}</tbody></table>`;
    $('growth-conclusion').textContent=r.runs.map(x=>`${names[x.type]}：${x.withoutIncomeBonusGain.low>0?'扣除收入加分后仍有可靠提升':'扣除收入加分后的提升尚不确定'}，胜率 ${pct(x.baseline.winRate)} → ${pct(x.candidate.winRate)}`).join('；')+'。得分提升不保证胜率同步提高；最大值是单局成绩，不能替代平均分或证明人类水平。';
    const labels={canalIndustry:'运河产业',canalLinks:'运河道路',railIndustry:'铁路产业',railLinks:'铁路道路'};
    $('growth-components').innerHTML=`<table class="ai-table"><thead><tr><th>平均分来自哪里？</th>${Object.values(labels).map(x=>`<th>${x} · 前 → 后</th>`).join('')}</tr></thead><tbody>${r.runs.map(x=>`<tr><td>${names[x.type]}</td>${Object.keys(labels).map(k=>`<td>${vp(x.baselineScores.components[k])} → ${vp(x.candidateScores.components[k])}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
}).catch(()=>{$('growth-caption').textContent='新一轮对局与训练进行中；完成独立复测后显示结果。';});
read('world_model/reports/score-training.json').then(r=>{
    const ci=x=>`${x.estimate>=0?'+':''}${x.estimate.toFixed(2)} [${x.low.toFixed(2)}, ${x.high.toFixed(2)}]`;
    $('score-caption').textContent=`以终局平均 VP 为主要标准 · 每种 AI 每版 ${r.runs[0].games} 局 · 固定三个对手，轮换座位`;
    $('score-table').innerHTML=`<table class="ai-table"><thead><tr><th>AI</th><th>续训前平均 VP</th><th>续训后平均 VP</th><th>分数变化 / 95% 区间</th><th>续训后胜率</th></tr></thead><tbody>${r.runs.map(x=>`<tr><td>${names[x.type]}</td><td>${vp(x.baseline.averageVP)}</td><td><strong>${vp(x.candidate.averageVP)}</strong></td><td>${ci(x.scoreGain)}</td><td>${pct(x.candidate.winRate)}</td></tr>`).join('')}</tbody></table>`;
    $('score-conclusion').textContent=r.runs.map(x=>`${names[x.type]}：${x.scoreGain.low>0?'区间支持平均分提高':x.scoreGain.high<0?'平均分下降':'尚不能确认平均分提高'}`).join('；')+'。胜率为辅助指标，模型选择依据开发赛分数，最终比赛使用新种子。';
}).catch(()=>{$('score-caption').textContent='分数导向续训正在进行；完成独立复测后显示结果。';});
read('world_model/reports/improvement.json').then(r=>{
    const ci=x=>`${(x.estimate*100).toFixed(1)} [${(x.low*100).toFixed(1)}, ${(x.high*100).toFixed(1)}]`;
    $('upgrade-caption').textContent=`每组 ${r.gamesPerRun} 局 · ${r.independentSeedGroups} 个全新种子组 · 深度 2、宽度 8 · 独立预测测试 ${r.prediction.testGames} 局`;
    $('upgrade-table').innerHTML=`<table class="ai-table"><thead><tr><th>实验阵容</th><th>搜索</th><th>学习增强搜索</th><th>神经网络</th><th>世界模型</th></tr></thead><tbody>${r.runs.map(run=>{const rate=k=>{const row=run.rows.find(x=>x.type===k);return row?pct(row.winRate):'未参赛';};return `<tr><td>${escapeHTML(run.name)}</td>${['search','guided','neural','world'].map(k=>`<td>${rate(k)}</td>`).join('')}</tr>`;}).join('')}</tbody></table>`;
    $('upgrade-conclusion').textContent=`新版世界模型 − 神经网络：${ci(r.worldMinusNeural)} 个百分点；学习增强搜索 − 原搜索：${ci(r.guidedMinusSearch)} 个百分点。方括号为 95% 区间。两套阵容的胜率不能直接横向排名。`;
}).catch(()=>{$('upgrade-caption').textContent='独立复测结果尚未生成。';});
read('world_model/reports/continuation.json').then(r=>{
    const signed=v=>`${v>=0?'+':''}${(v*100).toFixed(2)}`;
    $('continuation-caption').textContent=`每版 ${r.runs[0].completedGames} 局 · 相同 ${r.independentSeedGroups} 组种子与座位 · 原来的 100 局留出测试集`;
    $('continuation-table').innerHTML=`<table class="ai-table"><thead><tr><th>训练版本</th><th>世界模型胜率</th><th>神经网络胜率</th><th>差值 / 百分点（95% 区间）</th><th>测试 MSE</th><th>下一玩家准确率</th></tr></thead><tbody>${r.runs.map(run=>{const w=run.rows.find(x=>x.type==='world'),n=run.rows.find(x=>x.type==='neural'),ci=run.worldMinusNeuralWinRate;return `<tr><td>${escapeHTML(run.name)}</td><td>${pct(w.winRate)}</td><td>${pct(n.winRate)}</td><td>${signed(ci.estimate)} [${signed(ci.low)}, ${signed(ci.high)}]</td><td>${run.normalizedMSE.toFixed(6)}</td><td>${pct(run.currentPlayerAccuracy)}</td></tr>`;}).join('')}</tbody></table>`;
    const latest=r.runs.at(-1),gap=latest.worldMinusNeuralWinRate,change=latest.gapChangeVsOriginal;
    $('continuation-conclusion').textContent=`扩展数据后，世界模型与神经网络的胜率差为 ${signed(gap.estimate)} 个百分点。${gap.low>0?'本批对局的 95% 区间支持世界模型领先。':gap.high<0?'本批对局的 95% 区间支持神经网络领先。':'95% 区间包含 0，尚未拉开可靠差距。'}相对原版，差距变化 ${signed(change.estimate)} 个百分点，95% 区间 [${signed(change.low)}, ${signed(change.high)}]。`;
}).catch(()=>{$('continuation-caption').textContent='续训实验尚未完成，完成后会显示实际数据。';});
read('world_model/reports/learning-curve.json').then(r=>{
    $('curve-caption').textContent='初版的训练子集对照：相同验证集、24 轮训练；只改变训练游戏数量。10,000 局续训结果见上方“续训前后”。';
    $('learning-curve').innerHTML=`<table class="ai-table"><thead><tr><th>训练局数</th><th>训练样本</th><th>最佳验证 MSE</th></tr></thead><tbody>${r.runs.map(x=>`<tr><td>${x.trainingGames}</td><td>${x.trainingRows}</td><td>${x.validationMSE.toFixed(6)}</td></tr>`).join('')}</tbody></table>`;
}).catch(()=>{$('curve-caption').textContent='初版训练子集报告不可用。';});
