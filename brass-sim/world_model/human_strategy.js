(function(root) {
    'use strict';
    const node=typeof module!=='undefined'&&module.exports;
    const D=node?require('../js/gameData'):root;
    const clamp=(x,lo=0,hi=1)=>Math.max(lo,Math.min(hi,x));
    const sellable=type=>['cottonMill','manufacturer','pottery'].includes(type);
    const cityOf=key=>key.startsWith('farm:')?key.slice(5):key.slice(0,key.lastIndexOf('_'));

    function actionsLeft(state,player) {
        return state.players[player].hand.length+state.drawDeck.length/state.numPlayers;
    }

    function distances(state,start) {
        const distance=new Map([[start,0]]),queue=[start];
        while(queue.length) {
            queue.sort((a,b)=>distance.get(a)-distance.get(b));
            const city=queue.shift(),cost=distance.get(city);
            for(const link of D.CONNECTIONS) {
                if(!(state.era==='rail'?link.rail:link.canal)||!link.cities.includes(city))continue;
                const next=link.cities.find(c=>c!==city),n=cost+(state.boardLinks[link.id]?0:1);
                if(n<(distance.get(next)??Infinity)){distance.set(next,n);queue.push(next);}
            }
        }
        return distance;
    }

    function context(state) {
        const tiles=Object.entries(state.boardIndustries).map(([key,tile])=>({key,city:cityOf(key),tile}))
            .concat(Object.entries(state.breweryFarmTiles).filter(([,tile])=>tile).map(([city,tile])=>({key:'farm:'+city,city,tile})));
        const stock={coalMine:0,ironWorks:0,brewery:0};
        for(const {tile} of tiles)if(tile.type in stock)stock[tile.type]+=tile.resourceCubes||0;
        return {tiles,stock,distances:new Map()};
    }

    function sellProspect(state,player,city,type,beers,left,ctx) {
        const map=ctx.distances.get(city)||distances(state,city);ctx.distances.set(city,map);
        const merchants=state.merchantTiles.filter(m=>m.buys===null||m.buys===type);
        const distance=Math.min(...merchants.map(m=>map.get(m.location)??Infinity));
        if(!Number.isFinite(distance)||left<distance+1)return 0;
        const sources=state.findBeerSources(city,player);
        const available=sources.reduce((sum,s)=>sum+(s.type==='merchant'?
            +(merchants.some(m=>m===state.merchantTiles[s.index])):
            (s.key.startsWith('farm_')?state.breweryFarmTiles[s.key.slice(5)]:state.boardIndustries[s.key])?.resourceCubes||0),0);
        const missing=Math.max(0,beers-available);
        if(left<distance+1+Math.ceil(missing/2))return 0;
        // Connected beer is still contested; missing routes/beer require actions.
        return clamp((distance===0?.94:distance===1?.70:distance===2?.40:.12)-missing*.15);
    }

    function flipProspect(state,entry,left,ctx) {
        const {tile,city}=entry,p=tile.playerId;
        if(tile.flipped)return 1;
        if(left<1)return 0;
        if(sellable(tile.type))return sellProspect(state,p,city,tile.type,tile.tileData.beersToSell||0,left,ctx);
        const cubes=tile.resourceCubes||0,total=ctx.stock[tile.type]||cubes;
        if(tile.type==='brewery') {
            const ownDemand=ctx.tiles.filter(e=>e.tile.playerId===p&&!e.tile.flipped&&sellable(e.tile.type))
                .reduce((n,e)=>n+(e.tile.tileData.beersToSell||0),0);
            const rails=state.era==='rail'?Math.min(left/2,state.players[p].linksRemaining.rail/2):0;
            return clamp((ownDemand+rails+left*.12)/(Math.max(1,cubes)+total*.3),.05,.94);
        }
        // Coal/iron are support investments, not a substitute for VP industries.
        const marketDemand=tile.type==='ironWorks'?Math.max(0,8-state.ironMarket):Math.max(0,10-state.coalMarket);
        const connected=tile.type==='ironWorks'?1:
            [...state.getConnectedLocations(city)].some(D.isMerchantLocation)?1:.55;
        return clamp(connected*(left*(tile.type==='ironWorks'?.7:.8)+marketDemand)/Math.max(2,total+cubes),.03,.92);
    }

    function readiness(state,player,left) {
        if(left<3)return 0;
        const p=state.players[player],rail=state.era==='rail';
        const scores=[];
        for(const [type,stack] of Object.entries(p.industryTiles)) {
            const remaining=stack.filter(t=>!t.used);
            const market=sellable(type)?state.merchantTiles.some(m=>m.buys===null||m.buys===type):true;
            if(!market)continue;
            let best=0;
            for(let i=0;i<Math.min(remaining.length,6);i++) {
                const tile=remaining[i];
                if(rail?!tile.railEra:!tile.canalEra)continue;
                // A non-developable pottery must be built, not skipped for free.
                if(remaining.slice(0,i).some(t=>!t.canDevelop))break;
                const develop=Math.ceil(i/2),cycle=1+develop+(sellable(type)?.5:0)+(tile.cost>=14?.5:0);
                if(cycle>=left)continue;
                // Development pays back across repeated builds, but only while
                // enough actions remain to construct and sell the unlocked tiles.
                const production=remaining.slice(i).filter(t=>rail?t.railEra:t.canalEra).slice(0,3);
                const count=Math.min(production.length,Math.max(0,(left-develop)/2.5));
                const reward=production.reduce((n,t,j)=>n+t.vp*(!rail&&t.level>=2?2:1)*
                    Math.max(0,Math.min(1,count-j))*Math.pow(.75,j),0);
                const handAccess=p.hand.some(c=>c.type==='wildLocation'||c.type==='wildIndustry'||
                    c.type==='industry'&&c.industryType===type||c.type==='location'&&D.CITIES[c.location]?.slots.some(s=>s.includes(type)));
                best=Math.max(best,reward/cycle*Math.pow(.8,develop)*(handAccess?1:.65));
            }
            scores.push({type,score:best});
        }
        const products=scores.filter(s=>sellable(s.type)).sort((a,b)=>b.score-a.score);
        const supports=scores.filter(s=>!sellable(s.type)).sort((a,b)=>b.score-a.score);
        // A coherent industry path: don't reward developing every stack at once.
        return (products[0]?.score||0)*2.2+(supports[0]?.score||0)*.5;
    }

    function components(state,player,ctx=context(state)) {
        const p=state.players[player];
        if(state.gameOver)return {total:p.vp,settled:p.vp,industry:0,links:0,economy:0,readiness:0};
        const left=actionsLeft(state,player),canal=state.era==='canal';
        const probabilities=ctx.probabilities||(ctx.probabilities=new Map(ctx.tiles.map(e=>[e.key,flipProspect(state,e,actionsLeft(state,e.tile.playerId),ctx)])));
        let industry=0,links=0;
        for(const entry of ctx.tiles) {
            const t=entry.tile;
            if(t.playerId===player)industry+=t.tileData.vp*probabilities.get(entry.key)*
                (canal&&t.tileData.level>=2?1.9:1);
        }
        for(const [id,link] of Object.entries(state.boardLinks))if(link.playerId===player) {
            const cities=D.CONNECTIONS.find(c=>c.id===id).cities;
            links+=cities.filter(D.isMerchantLocation).length*2;
            for(const entry of ctx.tiles)if(cities.includes(entry.city))
                links+=(entry.tile.tileData.linkVP||0)*probabilities.get(entry.key);
        }
        const rounds=Math.max(0,(left-1)/2),income=state.getIncomeAmount(p.income);
        // Money is working capital. Its value approaches zero as actions run out.
        const usableCash=Math.min(p.money,Math.max(0,left*15));
        const nearCash=Math.min(usableCash,30),spareCash=Math.max(0,usableCash-30);
        const cashValue=(nearCash*.30+spareCash*.05)*clamp(left/3);
        const futureRounds=rounds+(canal?8:0);
        const incomeValue=Math.min(income*futureRounds,Math.max(0,left*12+ (canal?40:0)-p.money))*.16;
        const insolvency=Math.max(0,-income*rounds-p.money-30*Math.max(0,Math.floor((income+10)/3)))*.5;
        const economy=cashValue+incomeValue-insolvency;
        const development=readiness(state,player,left);
        return {settled:p.vp,industry,links,economy,readiness:development,
            total:p.vp+industry+links+economy+development};
    }

    function evaluate(state,player,{opponentWeight=.15}={}) {
        const ctx=context(state),values=state.players.map((_,p)=>components(state,p,ctx).total);
        return values[player]-opponentWeight*Math.max(...values.filter((_,p)=>p!==player));
    }

    const api={actionsLeft,distances,components,evaluate};
    if(node)module.exports=api;else root.BrassHumanStrategy=api;
})(globalThis);
