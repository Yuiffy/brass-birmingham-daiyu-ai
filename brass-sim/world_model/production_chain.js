(function(root) {
    'use strict';
    const node=typeof module!=='undefined'&&module.exports;
    const Sim=node?require('./simulator'):root.BrassSimulator;
    const Logic=node?require('../js/gameLogic'):root.GameLogic;
    const D=node?require('../js/gameData'):root;
    const Human=node?require('./human_strategy'):root.BrassHumanStrategy;
    const products=['cottonMill','manufacturer','pottery'];

    // Conservative own-action projection: known cards only, no future draw order,
    // no opponent actions or assumed income payments. All payments and consumption
    // delegate to the engine. It is a feasibility estimate, not a full game rollout.
    function discard(state,player,indices,{type,city}={}) {
        return indices.slice().sort((a,b)=>{
            const value=c=>!c?Infinity:c.type.startsWith('wild')?100:
                c.type==='location'&&c.location===city?80:
                c.type==='industry'&&c.industryType===type?75:
                c.type==='location'?40:30;
            return value(state.players[player].hand[a])-value(state.players[player].hand[b])||a-b;
        })[0];
    }
    function project(state,player,goal,{maxActions=6}={}) {
        let s=Sim.clone(state),trace=[],loans=0,spending=0;
        const budget=Math.min(maxActions,Math.floor(Human.actionsLeft(s,player)),s.players[player].hand.length);
        const act=(kind,target={})=>{
            if(trace.length>=budget)return false;
            const l=new Logic(s),p=s.players[player];
            const indices=l.getValidCardsForAction(player,kind,target);
            const cardIndex=discard(s,player,indices,goal);
            if(cardIndex===undefined)return false;
            const trial=Sim.clone(s),logic=new Logic(trial),before=trial.players[player].money;
            let result;
            if(kind==='loan')result=logic.executeLoan(player,cardIndex);
            else if(kind==='develop')result=logic.executeDevelop(player,target.type1,target.type2,cardIndex);
            else if(kind==='build')result=logic.executeBuild(player,target.cityId,target.slotIndex,target.industryType,cardIndex);
            else if(kind==='network')result=logic.executeNetwork(player,target.connectionId,cardIndex);
            else if(kind==='sell')result=logic.executeSell(player,target.keys,cardIndex);
            if(!result?.success)return false;
            const cost=before-trial.players[player].money;
            if(kind==='loan')loans++;else spending+=Math.max(0,cost);
            trace.push({action:kind,target,card:structuredClone(p.hand[cardIndex]),cost,
                money:trial.players[player].money,income:trial.getIncomeAmount(trial.players[player].income)});
            s=trial;return true;
        };
        // Query affordability using a separate copy. Resources, cards, slots and
        // network constraints remain real; imaginary cash is never executed.
        const funded=()=>{const t=Sim.clone(s);t.players[player].money=10000;return new Logic(t);};
        const afford=cost=>{
            while(s.players[player].money<cost)if(!act('loan'))return false;
            return true;
        };
        for(let remaining=goal.develop||0;remaining>0;) {
            const count=Math.min(2,remaining),iron=s.findIronSource(player).slice(0,count);
            if(iron.length<count||!afford(iron.reduce((n,x)=>n+(x.free?0:x.price),0)))return null;
            if(!act('develop',{type1:goal.type,type2:count===2?goal.type:null}))return null;
            remaining-=count;
        }
        let target=funded().getValidBuildTargets(player).find(t=>t.industryType===goal.type&&t.cityId===goal.city&&t.slotIndex===goal.slot);
        if(!target||!afford(target.cost.total))return null;
        target=new Logic(s).getValidBuildTargets(player).find(t=>t.industryType===goal.type&&t.cityId===goal.city&&t.slotIndex===goal.slot);
        if(!target||!act('build',target))return null;
        const key=`${goal.city}_${goal.slot}`,tile=s.boardIndustries[key];
        while(!new Logic(s).planSales(player,[key])) {
            const distances=Human.distances(s,goal.city);
            const merchants=s.merchantTiles.filter(m=>m.buys===null||m.buys===goal.type);
            const nearest=Math.min(...merchants.map(m=>distances.get(m.location)??Infinity));
            if(nearest>0) {
                const links=funded().getValidNetworkTargets(player).map(t=>{
                    const trial=Sim.clone(s);trial.boardLinks[t.connectionId]={playerId:player,type:s.era};
                    const d=Human.distances(trial,goal.city);
                    return {target:t,distance:Math.min(...merchants.map(m=>d.get(m.location)??Infinity))};
                }).filter(x=>x.distance<nearest).sort((a,b)=>a.distance-b.distance||a.target.cost-b.target.cost);
                if(!links.length||!afford(links[0].target.cost)||!act('network',links[0].target))return null;
            } else {
                const brewery=funded().getValidBuildTargets(player).filter(t=>t.industryType==='brewery')
                    .sort((a,b)=>a.cost.total-b.cost.total)[0];
                if(!brewery||!afford(brewery.cost.total)||!act('build',brewery))return null;
            }
        }
        const sales=new Logic(s).planSales(player,[key]);
        const contested=sales[0].beer.filter(b=>b.type!=='own').length;
        if(!act('sell',{keys:[key]}))return null;
        const gain=tile.tileData.vp*(s.era==='canal'&&tile.tileData.level>=2?1.9:1);
        return {goal,actions:trace.length,loans,spending,finalMoney:s.players[player].money,
            finalIncome:s.getIncomeAmount(s.players[player].income),gain,contestedBeer:contested,trace,
            score:gain/Math.max(2,trace.length)*Math.pow(.92,loans)*Math.pow(.85,contested)};
    }

    function plans(state,player,{maxActions=6,maxPlans=6}={}) {
        if(state.gameOver||state.players[player].hand.length<2)return [];
        const result=[];
        for(const type of products)for(const develop of [0,2]) {
            const trial=Sim.clone(state),p=trial.players[player];
            const remaining=p.industryTiles[type].filter(t=>!t.used);
            if(remaining.slice(0,develop).length!==develop||remaining.slice(0,develop).some(t=>!t.canDevelop))continue;
            remaining.slice(0,develop).forEach(t=>t.used=true);p.money=10000;
            const targets=new Logic(trial).getValidBuildTargets(player).filter(t=>t.industryType===type)
                .map(t=>({target:t,distance:Math.min(...trial.merchantTiles.filter(m=>m.buys===null||m.buys===type)
                    .map(m=>Human.distances(trial,t.cityId).get(m.location)??Infinity))}))
                .sort((a,b)=>a.distance-b.distance||a.target.cost.total-b.target.cost.total).slice(0,1);
            for(const {target} of targets) {
                const plan=project(state,player,{type,city:target.cityId,slot:target.slotIndex,develop},{maxActions});
                if(plan)result.push(plan);
            }
        }
        return result.sort((a,b)=>b.score-a.score).slice(0,maxPlans);
    }
    function components(state,player) {
        const base=Human.components(state,player);
        if(state.gameOver)return {...base,productionPlans:[]};
        const p=state.players[player],left=Human.actionsLeft(state,player),income=state.getIncomeAmount(p.income);
        // Cheap node evaluation. Full executable chains are computed once at the
        // root, where they can guide the first action without multiplying cost.
        const available=p.money+Math.max(0,income)*Math.max(0,(left-1)/2);
        const required=Math.min(60,Math.max(0,left-2)*8);
        const loans=Math.min(Math.max(0,Math.ceil((required-available)/30)),Math.max(0,Math.floor((income+10)/3)));
        const readiness=base.readiness*Math.max(0,1-loans/Math.max(1,left-2));
        // Loans consume actions and lower later income. Do not treat them as an
        // unlimited reserve when pricing forced liquidation risk.
        const shortage=Math.max(0,-income*Math.max(0,(left-1)/2)-p.money);
        const risk=shortage*.05;
        return {...base,readiness,economy:base.economy-risk,
            total:base.total-base.readiness+readiness-risk};
    }
    function evaluate(state,player,{opponentWeight=.15}={}) {
        const own=components(state,player).total;
        return own-opponentWeight*Math.max(...state.players.filter(p=>p.id!==player).map(p=>components(state,p.id).total));
    }
    const api={discard,project,plans,components,evaluate};
    if(node)module.exports=api;else root.BrassProductionChain=api;
})(globalThis);
