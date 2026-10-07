import { GOODS, PRICES, clone, rng, createGame, apply, legalActions, redeem, observation, wealth, capacity, actionLabel } from './engine.mjs';
import { policy, advice, arrivalProbabilities } from './ai.mjs';
const $ = id => document.getElementById(id);
const escape = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
let state, human = 0, worker = null, analysis = null, watching = false, watchTimer, revision = 0;
try { const saved = JSON.parse(localStorage.getItem('manila-v1')); state = validateSave(saved); } catch { state = createGame(); }
$('count').value = state.players.length; $('seed').value = state.seed;
const phases = { auction: '船长竞拍', setup: '购股与装载', placement: '部署帮手', roll: '掷骰航行', boarding: '海盗登船', 'pilot-small': '小领航员', 'pilot-large': '大领航员', 'pirate-route': '海盗处理货物', settlement: '收益结算', finished: '终局' };
function validateSave(s) {
  if (!s || s.version !== 1 || ![3, 4, 5].includes(s.players?.length) || !Number.isSafeInteger(s.seed) || !Number.isInteger(s.voyage) || !['auction','setup','placement','roll','boarding','pilot-small','pilot-large','pirate-route','settlement','finished'].includes(s.phase)) throw Error('存档格式不受支持');
  if (!Number.isInteger(s.actor) || s.actor < 0 || s.actor >= s.players.length || !Array.isArray(s.market) || s.market.length !== 4 || s.market.some(n => !Number.isInteger(n) || n < 0 || n > 5)) throw Error('存档状态无效');
  for (const p of s.players) {
    if (!Number.isFinite(p.cash) || p.cash < 0 || p.cash > 100000 || typeof p.name !== 'string') throw Error('存档玩家无效');
    for (const k of ['shares','mortgages','bought']) if (!Array.isArray(p[k]) || p[k].length !== 4 || p[k].some(n => !Number.isInteger(n) || n < 0 || n > 5)) throw Error('存档股票无效');
    if (p.shares.some((n,g) => n < p.mortgages[g] || n < p.bought[g]) || p.shares.reduce((a,b)=>a+b,0) - p.bought.reduce((a,b)=>a+b,0) !== 2) throw Error('存档持股不守恒');
  }
  if (!Array.isArray(s.supply) || s.supply.length !== 4 || s.supply.some((n,g) => n < 0 || n + s.players.reduce((v,p) => v+p.shares[g],0) !== 5)) throw Error('存档股票供应不守恒');
  if (!Array.isArray(s.log) || !Array.isArray(s.history) || !Array.isArray(s.boats) || s.boats.length > 3) throw Error('存档缺少航次信息');
  legalActions(s); return s;
}
function persist() { try { localStorage.setItem('manila-v1', JSON.stringify(state)); } catch {} }
function invalidate() { revision++; if (worker) worker.terminate(); worker = null; analysis = null; $('analysis-progress').textContent = ''; $('analysis-result').innerHTML = '<div class="empty-chart"><span>Δ</span><p>局面已更新。<br>在你的竞拍回合重新计算。</p></div>'; $('export-analysis').disabled = true; }
function play(a) {
  try { apply(state, a); invalidate(); persist(); render(); pumpOpponents(); }
  catch (e) { $('message').textContent = e.message; }
}
function pumpOpponents() {
  if (watching || state.phase === 'finished') return;
  let steps = 0;
  while (state.actor !== human && !['roll','settlement','finished'].includes(state.phase) && steps++ < 100) apply(state, policy(state, 'balanced', rng(state.seed + revision + steps)));
  persist(); render();
}
function button(label, action, classes = '') {
  const el = document.createElement('button'); el.className = classes; el.textContent = label; el.onclick = action; return el;
}
function render() {
  const finished = state.phase === 'finished';
  $('round-status').innerHTML = `第 <b>${state.voyage}</b> 航次 / ${phases[state.phase]}<br>${state.rolls} / 3 次航行 · ${state.players.length} 位商人`;
  $('market').innerHTML = GOODS.map((g, i) => `<div class="commodity"><strong><i class="dot" style="background:${g.color}"></i>${g.name}</strong><span class="price">${PRICES[state.market[i]]}<small>比索</small></span><div class="market-track">${PRICES.map((v,k) => `<span class="${k === state.market[i] ? 'current' : ''}">${v}</span>`).join('')}</div></div>`).join('');
  $('players').innerHTML = state.players.map((p, i) => {
    const shares = i === human || finished ? p.shares.map((n,g) => n ? `${GOODS[g].name} ×${n}` : '').filter(Boolean).join(' · ') : `初始 2 张保密${p.bought.some(n=>n) ? '<br>购入 ' + p.bought.map((n,g)=>n ? `${GOODS[g].name} ×${n}` : '').filter(Boolean).join(' · ') : ''}`;
    return `<div class="player ${state.actor === i && !finished ? 'active' : ''}"><b>${escape(p.name)}${state.captain === i ? ' / 船长' : ''}</b><div class="money">${p.cash}<small> 现金${finished ? ` · 财富 ${wealth(state,i)}` : ''}</small></div><div class="holdings">${shares}</div><div class="credit">抵押 ${p.mortgages.reduce((a,b)=>a+b,0)} 张${!finished ? ` · 帮手余 ${state.remaining[i]}` : ''}</div></div>`;
  }).join('');
  $('turn-title').textContent = finished ? '航海结束' : `${state.players[state.actor].name} · ${phases[state.phase]}`;
  $('turn-hint').textContent = watching ? '正在观察 AI 对局' : state.phase === 'placement' ? '停止后本航次不能再部署' : '你控制座位 1，其余为基线 AI';
  $('message').textContent = '';
  const controls = $('controls'); controls.replaceChildren();
  if (finished) {
    const ranking = state.players.map((p,i)=>({name:p.name,value:wealth(state,i)})).sort((a,b)=>b.value-a.value);
    const div=document.createElement('div'); div.className='final-score'; div.innerHTML = ranking.map((p,i)=>`${i+1}. ${escape(p.name)} <strong>${p.value}</strong> 比索`).join('<br>'); controls.append(div);
  } else if (watching) controls.append(button('暂停观察', stopWatching));
  else if (state.phase === 'setup' && state.actor === human) renderSetup(controls);
  else if (state.phase === 'auction' && state.actor === human) {
    const input = document.createElement('input'); input.type='number'; input.id='bid-amount'; input.setAttribute('aria-label','竞拍出价'); input.min=state.bid+1; input.max=Math.floor(capacity(state.players[human])); input.value=state.bid+1;
    controls.append(input, button('加价竞拍',()=>play({type:'bid', amount:Number(input.value)}),'primary'), button('放弃本轮',()=>play({type:'pass'})));
  } else {
    const actions = legalActions(state), ranked = advice(state);
    for (const a of actions) {
      const b = button(actionLabel(state,a), ()=>play(a), ['roll','settle'].includes(a.type) ? 'primary' : 'action');
      if (state.phase==='placement' && a.type==='place') { const score = ranked.find(r=>JSON.stringify(r.action)===JSON.stringify(a))?.score; if (score!==undefined) { const small=document.createElement('small'); small.textContent=`基线 ${score>=0?'+':''}${score.toFixed(1)}`; b.append(small); } }
      controls.append(b);
    }
  }
  if (!finished && !watching) {
    controls.append(button('AI 代走一步',()=>play(policy(state,'balanced',rng(state.seed+revision+701))),'quiet'));
    controls.append(button('观察 AI 全局',startWatching,'quiet'));
    for (let g=0;g<4;g++) if(state.players[human].mortgages[g] && state.players[human].cash>=15) controls.append(button(`赎回${GOODS[g].name} · 15`,()=>{try{redeem(state,human,g);invalidate();persist();render();}catch(e){$('message').textContent=e.message;}},'quiet'));
  }
  $('auction-status').innerHTML = state.phase==='auction' ? `<span>当前最高出价</span><br><strong>${state.bid}</strong> 比索${state.bidder!==null ? ` · ${escape(state.players[state.bidder].name)}` : ' · 尚未有人出价'}<br>你的支付能力 ${capacity(state.players[human])} · 现金 ${state.players[human].cash}` : `船长：${escape(state.players[state.captain].name)}<br>本航次成交 ${state.bidder===null?0:state.bid} 比索`;
  $('analyze').disabled = !!worker || watching || state.phase!=='auction' || state.actor!==human;
  $('log').replaceChildren(...state.log.slice(-20).reverse().map(text=>{const li=document.createElement('li');li.textContent=text;return li;}));
  draw();
}
function renderSetup(container) {
  const recommended=policy(state), form=document.createElement('div');form.className='setup';
  form.innerHTML=`<div class="setup-row"><label>留岸货物<select id="omit">${GOODS.map((g,i)=>`<option value="${i}" ${recommended.goods.includes(i)?'':'selected'}>${g.name}</option>`).join('')}</select></label><label>购入一股<select id="buy"><option value="-1">不购买</option>${GOODS.map((g,i)=>state.supply[i] ? `<option value="${i}" ${recommended.buy===i?'selected':''}>${g.name} · ${Math.max(5,PRICES[state.market[i]])} 比索</option>`:'').join('')}</select></label></div><div id="starts" class="setup-row"></div><div class="setup-row"><span id="sum" class="credit"></span></div>`;
  container.append(form);
  const update=()=>{
    const goods=GOODS.map((_,i)=>i).filter(i=>i!==Number($('omit').value));
    $('starts').innerHTML=goods.map((g,i)=>`<label>${GOODS[g].name}起点 <input class="start-input" data-good="${g}" aria-label="${GOODS[g].name}起点" type="number" min="0" max="5" value="${recommended.goods.includes(g)?recommended.starts[recommended.goods.indexOf(g)]:3}"></label>`).join('');
    document.querySelectorAll('.start-input').forEach(i=>i.oninput=sum);sum();
  };
  const sum=()=>{$('sum').textContent=`起点总和 ${[...document.querySelectorAll('.start-input')].reduce((v,i)=>v+Number(i.value),0)} / 9`;};
  $('omit').onchange=update;update();
  container.append(button('确认并开始部署',()=>{const inputs=[...document.querySelectorAll('.start-input')];play({type:'setup', buy:Number($('buy').value),goods:inputs.map(i=>Number(i.dataset.good)),starts:inputs.map(i=>Number(i.value))});},'primary'));
  container.append(button('采用 AI 装载方案',()=>play(recommended)));
}
function draw() {
  const c=$('board'), ctx=c.getContext('2d'), w=1080,h=540;c.width=w;c.height=h;
  ctx.fillStyle='#e5ece4';ctx.fillRect(0,0,w,h);
  ctx.strokeStyle='#cbd7ce';ctx.lineWidth=.6;
  for(let x=0;x<w;x+=36){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();}
  for(let y=0;y<h;y+=36){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();}
  ctx.fillStyle='#d4decf';ctx.beginPath();ctx.moveTo(970,0);ctx.bezierCurveTo(930,110,975,190,947,280);ctx.bezierCurveTo(926,360,980,430,945,540);ctx.lineTo(w,h);ctx.lineTo(w,0);ctx.fill();
  ctx.font='30px Georgia';ctx.fillStyle='#3c5656';ctx.fillText('MANILA',917,47);ctx.font='12px sans-serif';ctx.fillStyle='#677b72';ctx.fillText('目的港 / THE BLACK MARKET',855,70);
  ctx.font='12px Georgia';ctx.fillStyle='#7b8f83';ctx.fillText('CHART Nº 1821',30,35);
  const startX=99,dx=60;
  for(let lane=0;lane<3;lane++){
    const y=130+lane*118,b=state.boats[lane],g=GOODS[b?.good??lane];
    ctx.fillStyle=g.color;ctx.fillRect(22,y-23,4,46);ctx.font='15px sans-serif';ctx.fillText(g.name,34,y-1);ctx.font='10px sans-serif';ctx.fillStyle='#7b8f83';ctx.fillText(`${g.reward} 比索`,34,y+16);
    ctx.lineWidth=1;ctx.strokeStyle='#a9bbb0';ctx.beginPath();ctx.moveTo(startX,y);ctx.lineTo(940,y);ctx.stroke();
    for(let p=0;p<=13;p++){
      const x=startX+p*dx;ctx.strokeStyle=p<=5?'#819d8c':'#b1c2b6';ctx.beginPath();ctx.moveTo(x,y-7);ctx.lineTo(x,y+7);ctx.stroke();
      ctx.font='11px Georgia';ctx.textAlign='center';ctx.fillStyle=p===13?'#9a712d':'#7e9386';ctx.fillText(p,x,y+29);ctx.textAlign='left';
    }
    ctx.fillStyle='#8c9f90';ctx.font='10px sans-serif';ctx.fillText('港',935,y+29);
    const pos=b ? Math.min(14,b.pos):[2,3,4][lane],x=startX+pos*dx;
    ctx.save();ctx.translate(x,y);ctx.fillStyle=b?g.color:'#a0b4a4';ctx.strokeStyle='#f8f8ed';ctx.lineWidth=2;
    ctx.beginPath();ctx.moveTo(-27,-19);ctx.lineTo(24,-19);ctx.lineTo(35,0);ctx.lineTo(24,19);ctx.lineTo(-27,19);ctx.lineTo(-34,0);ctx.closePath();ctx.fill();ctx.stroke();
    ctx.fillStyle='#f5f3e8';ctx.font='13px Georgia';ctx.textAlign='center';ctx.fillText(b?String(b.pos):'—',0,4);
    if(b){ctx.font='10px sans-serif';ctx.fillStyle='#425f52';ctx.fillText(b.plundered?'遭劫掠':b.destination==='port'?'已到港':b.destination==='yard'?'进船厂':`${b.crew.length}/${g.costs.length} 帮手`,0,-29);}
    ctx.restore();
    if(b){ctx.font='10px sans-serif';ctx.fillStyle='#6d8477';ctx.fillText(`船员：${b.crew.length?b.crew.map(p=>state.players[p].name).join(' / '):'空船'}`,100,y+52);}
  }
  ctx.strokeStyle='#8b9d8e';ctx.setLineDash([3,5]);ctx.beginPath();ctx.moveTo(startX+13*dx,92);ctx.lineTo(startX+13*dx,425);ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle='#9a712d';ctx.font='10px sans-serif';ctx.fillText('13 / 海盗拦截线',819,441);
  ctx.fillStyle='#d9e3d7';ctx.fillRect(0,464,w,76);ctx.strokeStyle='#b8c9bb';ctx.beginPath();ctx.moveTo(0,464);ctx.lineTo(w,464);ctx.stroke();
  const groups=[['港口',state.port],['船厂',state.yard],['海盗',state.pirates],['领航',state.pilots],['保险',[state.insurer]]];
  groups.forEach(([name,list],i)=>{const x=29+i*211;ctx.font='13px sans-serif';ctx.fillStyle='#425f52';ctx.fillText(name,x,489);list.forEach((p,k)=>{ctx.fillStyle=p===null?'#b7c9b7':p===human?'#9a712d':'#526f64';ctx.beginPath();ctx.arc(x+12+k*48,514,9,0,Math.PI*2);ctx.fill();ctx.font='9px sans-serif';ctx.fillStyle='#f6f5ec';ctx.textAlign='center';ctx.fillText(p===null?'·':String(p+1),x+12+k*48,517);ctx.textAlign='left';});});
}
function stopWatching(){watching=false;clearTimeout(watchTimer);render();pumpOpponents();}
function startWatching(){invalidate();watching=true;render();watchTick();}
function watchTick(){if(!watching)return;if(state.phase==='finished'){watching=false;persist();render();return;}try{apply(state,policy(state,'balanced',rng(state.seed+revision++)));persist();render();watchTimer=setTimeout(watchTick,260);}catch(e){watching=false;render();$('message').textContent=e.message;}}
$('restart').onclick=()=>{watching=false;clearTimeout(watchTimer);invalidate();state=createGame(Number($('count').value),Number($('seed').value)||20261007);persist();render();};
$('rules-open').onclick=()=>$('rules').showModal();$('rules-close').onclick=()=>$('rules').close();
document.addEventListener('keydown',e=>{if(e.key==='f' && !['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName)){if(document.fullscreenElement)document.exitFullscreen();else document.documentElement.requestFullscreen().catch(()=>{});}});
$('analyze').onclick=()=>{
  if(worker)return;
  const snapshot=clone(state),targetRevision=revision;
  worker=new Worker(new URL('./worker.mjs',import.meta.url),{type:'module'});
  $('analyze').disabled=true;$('analysis-progress').textContent='抽样对手隐藏股票，开始配对模拟…';
  worker.onmessage=({data})=>{
    if(revision!==targetRevision)return;
    if(data.progress){$('analysis-progress').textContent=`已计算 ${data.progress.completed} / ${data.progress.total} 个价格`;return;}
    worker.terminate();worker=null;$('analyze').disabled=false;
    if(data.error){$('analysis-progress').textContent=data.error;return;}
    analysis={...data.result,context:{voyage:snapshot.voyage,observer:human,bid:snapshot.bid,revision:targetRevision,ownShares:snapshot.players[human].shares,market:snapshot.market}};
    renderAnalysis();$('export-analysis').disabled=false;
  };
  worker.onerror=e=>{if(worker)worker.terminate();worker=null;$('analyze').disabled=false;$('analysis-progress').textContent=`模拟失败：${e.message}`;};
  worker.postMessage({state:snapshot,observer:human,options:{samples:Number($('samples').value),horizon:Number($('horizon').value),maxBid:Number($('max-bid').value),style:$('style').value,seed:20261007}});
};
function renderAnalysis(){
  const a=analysis;if(!a.rows.length){$('analysis-result').textContent=a.reason;return;}
  $('analysis-progress').textContent=`${a.samples} 个配对样本 / ${a.horizon===0?'直至终局':a.horizon+' 航次'}`;
  const rows=a.rows,min=Math.min(-1,...rows.map(r=>r.lower)),max=Math.max(1,...rows.map(r=>r.upper)),W=280,H=145;
  const x=i=>32+i*(W-42)/Math.max(1,rows.length-1),y=v=>12+(max-v)/(max-min)*(H-35);
  const points=rows.map((r,i)=>`${x(i)},${y(r.delta)}`).join(' '),band=rows.map((r,i)=>`${x(i)},${y(r.upper)}`).concat(rows.map((r,i)=>`${x(i)},${y(r.lower)}`).reverse()).join(' ');
  const acceptable=a.meanAcceptableBids.length?a.meanAcceptableBids.join('、'):'无';
  $('analysis-result').innerHTML=`<div class="valuation"><div><span>均值盈亏平衡上限</span><strong>${a.breakEven} <small>比索</small></strong></div><div><span>谨慎上限 / 下界 ≥ 0</span><strong>${a.conservativeMax} <small>比索</small></strong></div></div><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="竞拍价格与相对退出的财富增益，阴影为均值近似95%置信区间"><line x1="30" x2="275" y1="${y(0)}" y2="${y(0)}" stroke="#a8b1a6" stroke-dasharray="3 3"/><polygon points="${band}" fill="#9a712d" opacity=".14"/><polyline points="${points}" fill="none" stroke="#9a712d" stroke-width="2"/><text x="0" y="${y(0)+3}" font-size="9" fill="#738080">0</text><text x="0" y="15" font-size="9" fill="#738080">+${max.toFixed(0)}</text><text x="0" y="${H-24}" font-size="9" fill="#738080">${min.toFixed(0)}</text><text x="32" y="${H-5}" font-size="10" fill="#738080">${rows[0].bid}</text><text x="250" y="${H-5}" font-size="10" fill="#738080">${rows.at(-1).bid} 比索</text></svg><p class="result-notes">均值可接受价格：${acceptable}。上限仅覆盖已测价格。${a.horizon===0?'终局财富':'短期市值，未计股票未来全部上涨'}。以下区间描述均值估计误差，单局波动更大。</p><details open><summary>每个价格的收益与误差</summary><table><thead><tr><th>出价</th><th>财富 Δ</th><th>95% 均值区间</th><th>${a.horizon===0?'胜率':'领先 Δ'}</th></tr></thead><tbody>${rows.map(r=>`<tr><td>${r.bid}</td><td>${r.delta>=0?'+':''}${r.delta.toFixed(1)}</td><td>${r.lower.toFixed(1)} ~ ${r.upper.toFixed(1)}</td><td>${r.winRate===null?r.marginDelta.toFixed(1):(r.winRate*100).toFixed(0)+'%'}</td></tr>`).join('')}</tbody></table></details><p class="result-notes">配对基准：退出后财富均值 ${a.baselineWealth.toFixed(1)}${a.baselineWin!==null?' / 终局胜率 '+(a.baselineWin*100).toFixed(0)+'%':''}。没有使用真实对手初始股票。模型：规则转移 + 基线策略，尚未训练。</p>`;
  $('analysis-result').querySelector('.valuation span').textContent = '已测均值可接受上限';
  if(a.rangeCensored){
    $('analysis-result').querySelector('.valuation strong').insertAdjacentText('afterbegin','≥ ');
    const note=document.createElement('p');note.className='result-notes';note.textContent='测试最高价格仍有正收益：尚未找到盈亏平衡点，请提高“出价测到”再计算。';$('analysis-result').prepend(note);
  }
}
function download(name,data){const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const el=document.createElement('a');el.href=url;el.download=name;el.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('save').onclick=()=>download(`manila-voyage-${state.voyage}.json`,state);
$('export-analysis').onclick=()=>{if(analysis)download(`manila-bid-analysis-${state.voyage}.json`,analysis);};
$('load').onclick=()=>$('load-file').click();$('load-file').onchange=async e=>{const file=e.target.files[0];if(!file)return;try{const imported=validateSave(JSON.parse(await file.text()));watching=false;clearTimeout(watchTimer);invalidate();state=imported;persist();render();pumpOpponents();}catch(e){$('message').textContent=`导入失败：${e.message}`;}e.target.value='';};
window.render_game_to_text=()=>JSON.stringify({coordinateSystem:'画布左上为原点；横向 0→13→港口；纵向三条航线',...observation(state,human),analysis:analysis?{breakEven:analysis.breakEven,conservativeMax:analysis.conservativeMax,horizon:analysis.horizon,samples:analysis.samples}:null,watching});
window.advanceTime=async ms=>{draw();};
render();pumpOpponents();
