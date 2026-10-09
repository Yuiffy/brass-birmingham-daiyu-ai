(function(root){
    'use strict';
    const Sim=BrassSimulator,E=BrassEncoding,P=BrassPlanner;
    const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    const fmt=n=>Number(n).toFixed(2);
    const api={world:null,worldByPlayers:{},guidedByPlayers:{},policyByPlayers:{},policy:null,error:null};
    let loading=null;
    api.load=()=>{
        if(api.policy&&[2,3,4].every(p=>api.worldByPlayers[p]&&api.policyByPlayers[p]))return Promise.resolve(true);
        if(loading)return loading;
        loading=(async()=>{
        try{
            const fetchModel=async(directory,file)=>{const r=await fetch(`${directory}/${file}`);if(!r.ok)throw Error(`${directory}/${file}: HTTP ${r.status}`);return new BrassWorldModel.Network(await r.json());};
            const [worlds,policy]=await Promise.all([
                Promise.all([2,3,4].map(async players=>[players,await fetchModel(`world_model/${BrassRuntimeModels.directoryForPlayers(players)}`,'world-model.json')])),
                fetchModel(`world_model/${BrassRuntimeModels.directoryForPlayers(4)}`,'neural-policy.json')
            ]);
            const guided=await Promise.all(worlds.map(async([players,world])=>{
                const directory=BrassRuntimeModels.guidedDirectoryForPlayers(players);
                return [players,directory===BrassRuntimeModels.directoryForPlayers(players)?world:await fetchModel(`world_model/${directory}`,'world-model.json')];
            }));
            const policies=await Promise.all([2,3,4].map(async players=>[players,
                E.schema.rulesVersion==='economy-v2'&&players!==4?await fetchModel(`world_model/${BrassRuntimeModels.directoryForPlayers(players)}`,'neural-policy.json'):policy]));
            api.policyByPlayers=Object.fromEntries(policies);
            api.guidedByPlayers=Object.fromEntries(guided);
            api.worldByPlayers=Object.fromEntries(worlds);api.world=api.worldByPlayers[4];api.policy=policy;api.error=null;
            document.querySelectorAll('option[value="guided"]').forEach(o=>o.disabled=!api.world.valueLayers);
            return true;
        }catch(e){api.error=e.message;return false;}finally{loading=null;}
        })();
        return loading;
    };
    let teacherLoading=null;
    api.loadTeacher=()=>{
        if(api.teacherWorld)return Promise.resolve(true);
        if(teacherLoading)return teacherLoading;
        teacherLoading=(async()=>{
            try {const url='world_model/experiments/human-teacher-20261009/models/world-model.json',r=await fetch(url);
                if(!r.ok)throw Error(`${url}: HTTP ${r.status}`);
                api.teacherWorld=new BrassWorldModel.Network(await r.json());api.teacherError=null;return true;
            }catch(e){api.teacherError=e.message;return false;}finally{teacherLoading=null;}
        })();return teacherLoading;
    };
    api.getWorld=(players,profile)=>profile==='teacher-trained-v2'?(players===4?api.teacherWorld||null:null):api.worldByPlayers[players]||null;
    api.getPolicy=players=>api.policyByPlayers[players]||null;
    api.getGuided=(players,profile)=>profile==='teacher-trained-v2'?api.getWorld(players,profile):api.guidedByPlayers[players]||api.getWorld(players);
    api.readyFor=(kinds,players=4)=>{
        if(!document.getElementById('ai-enabled').checked)return true;
        if(document.getElementById('guided-strategy')?.value==='teacher-trained-v2'&&kinds.some(t=>t==='world'||t==='guided')) {
            if(players!==4){document.getElementById('model-status').textContent='教师训练版目前仅支持四人局。';return false;}
            if(!api.teacherWorld){document.getElementById('model-status').textContent='教师训练权重尚未就绪。'+(api.teacherError||'');return false;}
        }
        const world=api.getWorld(players),missing=(kinds.includes('world')&&!world)||(kinds.includes('neural')&&!api.getPolicy(players));
        if((kinds.includes('guided')||kinds.includes('world'))&&!world){document.getElementById('model-status').textContent=`${players} 人模型尚未加载。`;return false;}
        if(kinds.includes('guided')&&!api.getGuided(players)?.valueLayers){document.getElementById('model-status').textContent='学习增强搜索需要已训练的局面估值权重。';return false;}
        if(missing){document.getElementById('model-status').textContent='训练权重尚不可用，请等待加载或改选搜索型 / 人类玩家。'+(api.error||'');return false;}
        return true;
    };
    api.setup=()=>{
        const enabled=document.getElementById('ai-enabled'), status=document.getElementById('model-status');
        const corrected=globalThis.BRASS_RULES==='economy-v2';
        const strategy=document.getElementById('guided-strategy');
        strategy.disabled=!corrected;
        const requested=new URLSearchParams(location.search).get('strategy');
        if(corrected&&['human-guide-v1','teacher-trained-v2'].includes(requested))strategy.value=requested;
        strategy.addEventListener('change',()=>{if(strategy.value==='teacher-trained-v2')api.loadTeacher().then(ok=>{
            status.textContent=ok?'四人教师训练权重已就绪（实验）。':`教师权重加载失败：${api.teacherError}`;
        });});
        if(strategy.value==='teacher-trained-v2')api.loadTeacher();
        document.getElementById('rules-status').textContent=corrected?'经济规则校准版：无终局收入加分，收入轨/贷款/时代/市场已修正；仍非完整官方规则，不能直接与真人比赛分数比较。':'历史训练版：保留旧规则与权重，含额外终局收入分。';
        if(corrected)document.querySelectorAll('a[href="arena.html"]').forEach(a=>a.href='arena.html?rules=economy-v2');
        const setEnabled=()=>{
            document.querySelectorAll('.player-ai-select').forEach(s=>{s.disabled=!enabled.checked;if(!enabled.checked)s.value='human';});
            if(enabled.checked){status.textContent='正在加载 2P / 3P / 4P 训练权重…';api.load().then(ok=>{status.textContent=ok?`模型已就绪 · 已按玩家人数加载 2P / 3P / 4P 动力学 · 4P 数据 ${api.world.data.datasetGames.toLocaleString()} 局 · 学习增强搜索可用`:`模型加载失败：${api.error}。搜索型和人类玩家仍可使用。`;});}
            else status.textContent='AI 模块已关闭 · 保留原游戏操作';
        };
        enabled.addEventListener('change',setEnabled);setEnabled();
        document.querySelectorAll('.count-btn').forEach(b=>b.addEventListener('click',()=>document.querySelectorAll('.player-ai-select').forEach(s=>s.disabled=!enabled.checked)));
        if(strategy.value==='teacher-trained-v2')document.querySelector('.count-btn[data-count="4"]')?.click();
        document.getElementById('ai-demo-btn').addEventListener('click',async()=>{
            const button=document.getElementById('ai-demo-btn');button.disabled=true;enabled.checked=true;
            try{
                const ok=await api.load();
                if(!ok){status.textContent=`模型加载失败：${api.error}`;return;}
                const selects=[...document.querySelectorAll('.player-ai-select')];
                const kinds=selects.map((s,i)=>{const type=i%2?'world':'guided';s.disabled=false;s.value=type;document.querySelectorAll('.player-name-input input')[i].value=P.TYPES[type];return type;});
                if(api.readyFor(kinds,selects.length))document.getElementById('start-game-btn').click();
            }finally{button.disabled=false;}
        });
    };
    class Controller {
        constructor(state,ui,kinds){
            this.state=state;this.ui=ui;this.kinds=state.players.map((_,i)=>kinds[i]||'human');
            this.profile=state.rulesVersion==='economy-v2'?document.getElementById('guided-strategy')?.value:null;
            this.strategy=this.profile==='teacher-trained-v2'?'human-card-v2':this.profile==='human-guide-v1'?'human-guide-v1':null;
            this.running=true;this.timer=null;this.disposed=false;this.selection=null;
            const panel=document.createElement('div');panel.id='ai-toolbar';
            panel.innerHTML='<span id="ai-turn-label"></span><button class="ai-button" id="ai-pause">暂停 AI</button><button class="ai-button" id="ai-step">AI 走一步</button><button class="ai-button" id="ai-inspect">AI 推演 / 预测对照</button><a href="arena.html" target="_blank" rel="noopener">对战实验室 ↗</a>';
            document.getElementById('game-screen').prepend(panel);this.toolbar=panel;
            if(globalThis.BRASS_RULES==='economy-v2')panel.querySelector('a').href='arena.html?rules=economy-v2';
            const dialog=document.createElement('dialog');dialog.className='imagination-dialog';dialog.id='imagination-dialog';
            dialog.innerHTML='<div class="imagination-header"><div><small>BRASS · AI IMAGINATION</small><h2>在模型中想象未来</h2></div><button id="ai-close" class="ai-button">关闭</button></div><p class="ai-note">世界模型预测会出错。右侧真实结果由原规则引擎在副本上计算；只有点击执行才会改变棋局。</p><div class="ai-controls"><label>推演深度 <select id="ai-depth"><option>1</option><option>2</option><option selected>3</option><option>5</option></select></label><button class="ai-button" id="ai-rethink">重新推演</button><span id="ai-plan-info" role="status"></span></div><div class="imagination-grid"><div><h3>候选动作与未来评分</h3><div id="ai-candidates"></div></div><div><h3>Prediction vs Reality</h3><div id="ai-comparison">选择左侧动作查看。</div></div></div>';
            document.body.appendChild(dialog);this.dialog=dialog;
            document.getElementById('ai-close').onclick=()=>dialog.close();
            dialog.addEventListener('close',()=>this.refresh());
            document.getElementById('ai-pause').onclick=()=>{this.running=!this.running;this.refresh();};
            document.getElementById('ai-step').onclick=()=>{this.running=false;this.move();};
            document.getElementById('ai-inspect').onclick=()=>{this.running=false;this.refresh();dialog.showModal();this.inspect();};
            document.getElementById('ai-rethink').onclick=()=>this.inspect();
            this.originalRefresh=ui.refresh.bind(ui);ui.refresh=()=>{this.originalRefresh();this.refresh();};
            this.originalSelect=ui.onActionSelected.bind(ui);ui.onActionSelected=a=>{if(this.kinds[state.currentPlayerId]==='human')this.originalSelect(a);};
            this.refresh();
        }
        dispose(){this.disposed=true;clearTimeout(this.timer);this.toolbar.remove();this.dialog.remove();document.getElementById('game-screen').classList.remove('ai-turn');this.ui.refresh=this.originalRefresh;this.ui.onActionSelected=this.originalSelect;}
        refresh(){
            clearTimeout(this.timer);if(this.disposed)return;
            const type=this.kinds[this.state.currentPlayerId];
            const active=type!=='human'&&!this.state.gameOver;
            document.getElementById('game-screen').classList.toggle('ai-turn',active);
            document.getElementById('ai-turn-label').textContent=`${this.state.currentPlayer.name} · ${P.TYPES[type]}${this.profile==='teacher-trained-v2'?' · 教师训练':type==='guided'&&this.strategy?' · 攻略增强':''}${this.state.gameOver?' · 已结束':''}`;
            document.getElementById('ai-pause').textContent=this.running?'暂停 AI':'继续 AI';
            document.getElementById('ai-step').disabled=!active;
            document.getElementById('ai-inspect').disabled=this.state.gameOver||!api.getWorld(this.state.numPlayers,this.profile);
            if(active&&this.running&&!this.dialog.open&&!this.state.gameOver&&document.getElementById('scoring-overlay').classList.contains('hidden'))this.timer=setTimeout(()=>this.move(),650);
        }
        move(){
            if(this.disposed||this.state.gameOver)return;
            const type=this.kinds[this.state.currentPlayerId];if(type==='human')return;
            try{const result=P.plan(this.state,this.planOptions(type,2));this.lastPlan=result;this.apply(result.selected);}
            catch(e){this.running=false;this.ui.showToast(e.message,'error');this.refresh();}
        }
        planOptions(type,depth){
            return {type,world:type==='guided'?api.getGuided(this.state.numPlayers,this.profile):api.getWorld(this.state.numPlayers,this.profile),
                policy:api.getPolicy(this.state.numPlayers),depth,width:8,strategy:type==='guided'?this.strategy||null:null};
        }
        apply(action){
            const id=this.state.currentPlayerId,era=this.state.era;
            const result=Sim.step(this.state,action);
            Object.assign(this.state,Sim.snapshot(result.state));
            this.ui.cancelAction();this.ui.addLogEntry(id,`${P.TYPES[this.kinds[id]]}: ${result.result.message}`,'action');
            if(result.event.type==='era'){this.ui.addLogEntry(null,`${era} 结算 → 铁路时代`,'era');this.ui.showScoring('Canal Era Complete',result.event.scores);}
            if(result.event.type==='game')this.ui.showGameOver(result.event.scores);
            this.ui.refresh();
        }
        inspect(){
            const world=api.getWorld(this.state.numPlayers,this.profile);if(!world||this.state.gameOver)return;
            this.previewToken=JSON.stringify(Sim.snapshot(this.state));
            try{
                const depth=Number(document.getElementById('ai-depth').value);
                const active=this.kinds[this.state.currentPlayerId],type=active==='human'?'world':active;
                const result=P.plan(this.state,this.planOptions(type,depth));this.preview=result;
                this.dialog?.querySelector('h2')?.replaceChildren(document.createTextNode(type==='guided'||type==='search'?'规则推演与局面估值':'在模型中想象未来'));
                document.getElementById('ai-plan-info').textContent=`${P.TYPES[type]}${this.profile==='teacher-trained-v2'?' · 教师训练＋保牌':result.strategy?' · 攻略增强':''} · 预比较 ${result.candidatePoolSize} 个动作 · ${depth} 步上限 · ${fmt(result.elapsedMs)} ms`;
                document.getElementById('ai-candidates').innerHTML=result.branches.map((b,i)=>`<button class="candidate-button ${i===0?'best':''}" data-index="${i}"><span>${i===0?'推荐 · ':''}${esc(b.label)}</span><b>${fmt(b.score)}</b><small>${b.steps.map(s=>esc(s.label)).join(' → ')}${b.steps.length<depth?' · 预测状态无后续候选，提前停止':''}</small></button>`).join('');
                document.querySelectorAll('.candidate-button').forEach(b=>b.onclick=()=>this.compare(Number(b.dataset.index)));
                if(result.branches.length)this.compare(0);
            }catch(e){document.getElementById('ai-plan-info').textContent=e.message;}
        }
        compare(index){
            if(this.previewToken!==JSON.stringify(Sim.snapshot(this.state))){document.getElementById('ai-comparison').textContent='棋局已变化，请重新推演。';return;}
            const branch=this.preview.branches[index],before=E.encodeState(this.state),action=branch.action;
            const world=api.getWorld(this.state.numPlayers,this.profile);
            if(action.target?.connectionIds&&world.data?.actionEncoding!=='resource-network-v2'){
                const truth=E.encodeState(Sim.step(this.state,action).state);
                const rows=E.fields.map((f,i)=>({name:f.name,b:before[i]*f.scale,t:truth[i]*f.scale})).filter(r=>Math.abs(r.t-r.b)>.1);
                document.getElementById('ai-comparison').innerHTML=`<p><strong>${esc(branch.label)}</strong></p><p class="ai-note" id="ai-truth-status">规则引擎预览（尚未执行）。现有世界模型未训练双铁路，本动作展示真实模拟变化。</p><button class="ai-button primary" id="ai-execute">执行真实动作</button><div class="comparison-scroll"><table class="ai-table"><thead><tr><th>字段</th><th>当前</th><th>真实结果</th></tr></thead><tbody>${rows.map(r=>`<tr><td>${esc(r.name)}</td><td>${fmt(r.b)}</td><td>${fmt(r.t)}</td></tr>`).join('')}</tbody></table></div>`;
                this.bindExecute(action);return;
            }
            const predicted=world.predict(before,E.encodeAction(action,this.state,{version:world.data?.actionEncoding||'legacy'})),truth=E.encodeState(Sim.step(this.state,action).state);
            const rows=E.fields.map((f,i)=>({name:f.name,b:before[i]*f.scale,p:predicted[i]*f.scale,t:truth[i]*f.scale,group:f.group}))
                .filter(r=>Math.abs(r.t-r.b)>.1||Math.abs(r.p-r.b)>.5)
                .sort((a,b)=>{const priority={money:0,income:1,vp:2,currentPlayer:3};return (priority[a.group]??4)-(priority[b.group]??4)||Math.abs(b.t-b.p)-Math.abs(a.t-a.p);});
            let actualState=Sim.clone(this.state),imagined=before.slice(),nextAction=action,errors=[];
            for(let d=1;d<=5;d++){
                const encodedAction=E.encodeAction(nextAction,actualState,{version:world.data?.actionEncoding||'legacy'});
                imagined=world.predict(imagined,encodedAction);
                actualState=Sim.step(actualState,nextAction).state;
                const expected=E.encodeState(actualState),mae=expected.reduce((sum,v,i)=>sum+Math.abs(v-imagined[i]),0)/expected.length;
                errors.push({d,mae});if(actualState.gameOver)break;
                nextAction=Sim.candidates(actualState).sort((a,b)=>b.score-a.score)[0];if(!nextAction)break;
            }
            const max=Math.max(...errors.map(e=>e.mae),0.001);
            document.getElementById('ai-comparison').innerHTML=`<p><strong>${esc(branch.label)}</strong></p><p class="ai-note" id="ai-truth-status">真实模拟器预览（尚未执行）· 展示变化或预测变化的字段 ${rows.length} 项</p><button class="ai-button primary" id="ai-execute">执行真实动作并比较</button><div class="comparison-scroll"><table class="ai-table"><thead><tr><th>字段</th><th>当前</th><th>模型预测</th><th>真实结果</th><th>取整匹配</th></tr></thead><tbody>${rows.map(r=>`<tr><td>${esc(r.name)}</td><td>${fmt(r.b)}</td><td>${fmt(r.p)}</td><td>${fmt(r.t)}</td><td class="${Math.round(r.p)===Math.round(r.t)?'match':'mismatch'}">${Math.round(r.p)===Math.round(r.t)?'✓':'✗'}</td></tr>`).join('')}</tbody></table></div><h3>递归预测：误差如何累积</h3><p class="ai-note">固定同一段真实合法动作序列，模型连续预测，中间不校正。下列数值为标准化 MAE；与左侧自由规划分支分开评估。</p><div class="rollout-bars">${errors.map(e=>`<div><span>${e.d} 步</span><i style="width:${e.mae/max*65}%"></i><b>${e.mae.toFixed(4)}</b></div>`).join('')}</div>`;
            this.bindExecute(action);
        }
        bindExecute(action){
            document.getElementById('ai-execute').onclick=()=>{
                if(this.previewToken!==JSON.stringify(Sim.snapshot(this.state))){document.getElementById('ai-truth-status').textContent='棋局已变化，请重新推演。';return;}
                try{this.apply(action);document.getElementById('ai-truth-status').textContent='已执行真实动作 · 表中真实结果与当前执行结果一致';document.getElementById('ai-execute').disabled=true;}
                catch(e){document.getElementById('ai-truth-status').textContent=e.message;}
            };
        }
    }
    api.Controller=Controller;root.BrassAI=api;
})(globalThis);
