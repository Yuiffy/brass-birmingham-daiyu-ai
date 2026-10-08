(function(root){
    'use strict';
    const node=typeof module!=='undefined'&&module.exports;
    const Sim=node?require('./simulator'):root.BrassSimulator;
    const Planner=node?require('./planner'):root.BrassPlanner;
    const E_rulesVersion=()=>node?require('../js/gameData').RULES_VERSION:root.RULES_VERSION;
    async function tournament({games=12,seed=800001,depth=2,width=8,types=['heuristic','search','neural','world'],world,policy,guidedWorld,onProgress=()=>{},cancelled=()=>false}={}) {
        if(!Number.isInteger(games)||games<1||!Array.isArray(types)||types.length<2||types.length>4)throw Error('Invalid tournament configuration');
        if(types.some(t=>!['heuristic','search','neural','world','guided'].includes(t)))throw Error('Invalid AI entrant');
        if(types.includes('world')&&!world||types.includes('neural')&&!policy)throw Error('Load trained models before comparison');
        if(types.includes('guided')&&!(guidedWorld||world)?.valueLayers)throw Error('Load a learned value model before guided search');
        const totals=Object.fromEntries(types.map(t=>[t,{type:t,games:0,wins:0,vp:0,moves:0,thinkingMs:0}]));
        const records=[];
        for(let g=0;g<games;g++) {
            if(cancelled())break;
            const seats=types.map((_,i)=>types[(i+g)%types.length]);
            const gameSeed=seed+Math.floor(g/types.length)*9973;
            let state=Sim.create(types.length,gameSeed,seats.map(t=>Planner.TYPES[t])),moves=0;
            const components=seats.map(()=>({canalIndustry:0,canalLinks:0,railIndustry:0,railLinks:0,actions:{}}));
            while(!state.gameOver) {
                if(cancelled())break;
                if(++moves>1000)throw Error('Tournament failed to terminate');
                const type=seats[state.currentPlayerId];
                const result=Planner.plan(state,{type,world:type==='guided'?(guidedWorld||world):world,policy,depth,width});
                if(!result.selected)throw Error('AI produced no action');
                totals[type].moves++;totals[type].thinkingMs+=result.elapsedMs;
                const actor=state.currentPlayerId,action=result.selected.action,era=state.era;
                components[actor].actions[action]=(components[actor].actions[action]||0)+1;
                const stepped=Sim.step(state,result.selected);state=stepped.state;
                if(stepped.event?.scores)for(const score of stepped.event.scores){
                    components[score.playerId][`${era}Industry`]+=score.industryVP;
                    components[score.playerId][`${era}Links`]+=score.linkVP;
                }
                if(moves%8===0) {
                    onProgress({game:g+1,games,moves,era:state.era,round:state.round});
                    await new Promise(resolve=>setTimeout(resolve,0));
                }
            }
            if(!state.gameOver)break;
            const best=state.players.slice().sort((a,b)=>state.comparePlayers(a,b))[0],winners=state.players.filter(p=>state.comparePlayers(p,best)===0);
            const max=Math.max(...state.players.map(p=>p.vp));
            const scores=state.players.map((p,i)=>({seat:i,type:seats[i],vp:p.vp,incomeBonus:state.rulesVersion==='economy-v2'?0:p.income, incomeLevel:state.getIncomeAmount(p.income), money:p.money,
                vpWithoutIncomeBonus:p.vp-(state.rulesVersion==='economy-v2'?0:p.income),components:components[i]}));
            scores.forEach(p=>{const t=totals[p.type];t.games++;t.vp+=p.vp;if(winners.some(w=>w.id===p.seat))t.wins+=1/winners.length;});
            records.push({game:g+1,seed:gameSeed,seats,scores,moves});
            onProgress({game:g+1,games,completed:true,scores});
        }
        const describe=network=>network?.data?{kind:network.data.kind,epoch:network.data.epoch,
            datasetGames:network.data.datasetGames,trainingRows:network.data.trainingRows,
            planningValue:network.data.planningValue||'heuristic',valueEpoch:network.data.valueModel?.epoch,
            valueObjective:network.data.valueModel?.target,valueFeatures:network.data.valueModel?.featureVersion,
            shortlist:network.data.planningShortlist||'legacy',candidateSelection:network.data.planningCandidates||'legacy',
            continuation:network.data.planningContinuation||'legacy'}:null;
        return {config:{games,players:types.length,seed,depth,width,types,rulesVersion:E_rulesVersion()},models:{world:describe(world),guided:describe(guidedWorld||world),policy:describe(policy)},completedGames:records.length,
            cancelled:records.length<games,balancedSeats:records.length%types.length===0,
            rows:Object.values(totals).map(t=>({...t,averageVP:t.games?t.vp/t.games:0,winRate:t.games?t.wins/t.games:0,
                millisecondsPerMove:t.moves?t.thinkingMs/t.moves:0})),games:records,
            notes:['Same seed reused across a complete rotation of seats; ties share one win. With repeated types, row.games counts player appearances and winRate is per player appearance.',
                'Search uses original engine transitions with sampled unknown future cards. World model uses learned transitions only.',
                'Depth and retained width are shared. learned-pool-v1 ranks up to 32 roots; diverse-pool-12 considers up to 12 subsequent actions instead of 3. Wall-clock cost is reported, not equalized.',
                E_rulesVersion()==='economy-v2'?'Economy-v2: no terminal income bonus; official economy/era corrections only, remaining engine limitations are documented.':'Legacy VP includes an unofficial terminal income bonus; vpWithoutIncomeBonus removes only that bonus.',
                'Classic search uses heuristic state evaluation. Guided search uses learned terminal value with real engine transitions. World uses the value function recorded in model metadata.',
                'Results measure this implementation, observation setting, opponent lineup and compute budget; they do not establish general architectural superiority.']};
    }
    if(node)module.exports={tournament};else root.BrassTournament={tournament};
})(globalThis);
