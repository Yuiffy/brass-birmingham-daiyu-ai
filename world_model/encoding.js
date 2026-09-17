(function (root) {
    'use strict';
    const node = typeof module !== 'undefined' && module.exports;
    const D = node ? require('../js/gameData') : root;
    const Sim = node ? require('./simulator') : root.BrassSimulator;
    const types = Object.values(D.INDUSTRY_TYPES).sort();
    const cities = Object.keys(D.CITIES).sort();
    const slots = cities.flatMap(c => D.CITIES[c].slots.map((_, i) => `${c}_${i}`))
        .concat(['farm:northern', 'farm:southern']);
    const links = D.CONNECTIONS.map(c => c.id).sort();
    const actions = Object.values(D.ACTIONS);
    const cards = cities.map(location => ({ type: 'location', location, name: D.CITIES[location].name }))
        .concat(types.map(industryType => ({ type: 'industry', industryType, name: D.INDUSTRY_DISPLAY[industryType].name })),
            [{ type: 'wildLocation', name: 'Wild Location' }, { type: 'wildIndustry', name: 'Wild Industry' }]);
    const cardKey = c => [c.type, c.location || c.industryType || ''].join(':');
    const cardIds = cards.map(cardKey);
    const merchants = [2, 3, 4].flatMap(n => D.MERCHANT_TILES[n] || [])
        .map(m => `${m.location}:${m.buys || '*'}`).sort();
    const fields = [];
    const field = (name, group, scale = 1) => fields.push({ name, group, scale });
    field('era', 'era'); field('round', 'turn', 16); field('players', 'turn', 4);
    for (let p = 0; p < 4; p++) field(`current.${p}`, 'currentPlayer');
    field('actionsThisTurn', 'turn', 2); field('actionsPerTurn', 'turn', 2);
    field('isFirstRound', 'turn'); field('gameOver', 'turn');
    field('deckSize', 'cards', 64); field('wildLocationPile', 'cards', 4); field('wildIndustryPile', 'cards', 4);
    field('coalMarket', 'coalMarket', 14); field('ironMarket', 'ironMarket', 10);
    for (let p = 0; p < 4; p++) field(`order.${p}`, 'turn', 4);
    for (let p = 0; p < 4; p++) {
        for (const [k, group, scale] of [['money','money',100], ['income','income',30], ['vp','vp',100],
            ['handSize','cards',8], ['spent','turn',50], ['canal','network',14], ['rail','network',14],
            ['wildLocation','cards',1], ['wildIndustry','cards',1]]) field(`p${p}.${k}`, group, scale);
        for (const t of types) field(`p${p}.used.${t}`, 'supply', 12);
        for (const id of cardIds) field(`p${p}.card.${id}`, 'cards', 4);
    }
    for (const slot of slots) {
        for (const [k, group, scale] of [['owner','industry',4], ['type','industry',6], ['level','industry',8],
            ['flipped','flip',1], ['cubes','resources',8], ['vp','industry',20],
            ['income','industry',12], ['linkVP','industry',3]]) field(`slot.${slot}.${k}`, group, scale);
    }
    for (const link of links) { field(`link.${link}.owner`, 'network', 4); field(`link.${link}.rail`, 'network'); }
    merchants.forEach((m, i) => ['present','beer','claimed'].forEach(k => field(`merchant.${i}.${m}.${k}`, 'merchant')));
    const indices = Object.fromEntries(fields.map((f, i) => [f.name, i]));
    const schema = { version: 'brass-wm-v1', observation: 'all-hands-public-board-no-future-deck-order', fields,
        types, cities, slots, links, cards, merchants, actions };

    function encodeState(s) {
        const v = [], add = x => v.push(Number(x) || 0);
        add(s.era === 'rail'); add(s.round); add(s.numPlayers);
        for (let p = 0; p < 4; p++) add(s.currentPlayerId === p);
        [s.actionsThisTurn,s.actionsPerTurn,s.isFirstRound,s.gameOver,s.drawDeck.length,
            s.wildLocationPile,s.wildIndustryPile,s.coalMarket,s.ironMarket].forEach(add);
        for (let p = 0; p < 4; p++) add(s.turnOrder[p] === undefined ? 0 : s.turnOrder[p] + 1);
        for (let p = 0; p < 4; p++) {
            const pl = s.players[p];
            [pl?.money,pl?.income,pl?.vp,pl?.hand.length,s.moneySpentThisRound[p],
                pl?.linksRemaining.canal,pl?.linksRemaining.rail,pl?.hasWildLocation,pl?.hasWildIndustry].forEach(add);
            types.forEach(t => add(pl?.industryTiles[t].filter(t => t.used).length));
            cardIds.forEach(id => add(pl?.hand.filter(c => cardKey(c) === id).length));
        }
        slots.forEach(slot => {
            const t = slot.startsWith('farm:') ? s.breweryFarmTiles[slot.slice(5)] : s.boardIndustries[slot];
            [t ? t.playerId + 1 : 0,t ? types.indexOf(t.type) + 1 : 0,t?.tileData.level,t?.flipped,
                t?.resourceCubes,t?.tileData.vp,t?.tileData.income,t?.tileData.linkVP].forEach(add);
        });
        links.forEach(id => { const l = s.boardLinks[id]; add(l ? l.playerId + 1 : 0); add(l?.type === 'rail'); });
        const used = new Set();
        merchants.forEach(id => {
            const i = s.merchantTiles.findIndex((m, i) => !used.has(i) && `${m.location}:${m.buys || '*'}` === id);
            if (i >= 0) used.add(i);
            add(i >= 0); add(s.merchantTiles[i]?.hasBeer); add(s.merchantTiles[i]?.bonusClaimed);
        });
        if (v.length !== fields.length) throw Error('State schema dimension mismatch');
        return v.map((x, i) => x / fields[i].scale);
    }

    function encodeAction(a, s) {
        const t = a.target || {}, tile = t.tileData || t.tile?.tileData || {};
        const v = actions.map(x => +(a.action === x));
        for (let p = 0; p < 4; p++) v.push(+(s.currentPlayerId === p));
        const slotKey = t.key || `${t.cityId}_${t.slotIndex}`;
        v.push(...slots.map(x => +(x === slotKey)), ...links.map(x => +(x === t.connectionId)));
        v.push(...types.map(x => +(x === (t.industryType || t.type1 || t.tile?.type))),
            ...types.map(x => +(x === t.type2)));
        const discarded = (a.cardIndices || [a.cardIndex]).map(i => s.currentPlayer.hand[i]).filter(Boolean);
        v.push(...cardIds.map(id => discarded.filter(c => cardKey(c) === id).length / 3));
        v.push((t.cost?.total ?? t.cost ?? 0) / 50, (tile.level || 0) / 8,
            (tile.vp || 0) / 20, (tile.income || 0) / 12, (tile.costCoal || 0) / 3,
            (tile.costIron || 0) / 3, (tile.resourceCubes || 0) / 8, (tile.beersToSell || 0) / 3);
        return v;
    }
    const raw = (v, name) => v[indices[name]] * fields[indices[name]].scale;
    // Approximate projection for proposing the next imagined action only.
    // This never runs an engine transition or repairs a prediction using truth.
    function decodeState(v, base, {structured=false}={}) {
        const s = Sim.clone(base), get = name => Math.round(raw(v, name)), bounded = (n,a,b) => Math.max(a,Math.min(b,n));
        s.era = get('era') ? 'rail' : 'canal'; s.round = Math.max(1,get('round'));
        s.actionsThisTurn = bounded(get('actionsThisTurn'),0,1); s.actionsPerTurn = bounded(get('actionsPerTurn'),1,2);
        s.isFirstRound = !!get('isFirstRound'); s.gameOver = !!get('gameOver');
        if(structured){
            const proposed=Array.from({length:s.numPlayers},(_,p)=>get(`order.${p}`)-1);
            s.turnOrder=[...new Set(proposed.filter(p=>p>=0&&p<s.numPlayers))];
            for(let p=0;p<s.numPlayers;p++)if(!s.turnOrder.includes(p))s.turnOrder.push(p);
            // The future deck is not observable: only its predicted size is used.
            const placeholder=cards.find(c=>c.type==='location');
            s.drawDeck=Array.from({length:bounded(get('deckSize'),0,64)},()=>({...placeholder}));
        }
        const actor = Array.from({length:s.numPlayers}, (_,p) => [p,raw(v,`current.${p}`)]).sort((a,b)=>b[1]-a[1])[0][0];
        s.currentPlayerIndex = s.turnOrder.indexOf(actor);
        s.coalMarket = bounded(get('coalMarket'),0,14); s.ironMarket = bounded(get('ironMarket'),0,10);
        s.wildLocationPile = bounded(get('wildLocationPile'),0,4); s.wildIndustryPile = bounded(get('wildIndustryPile'),0,4);
        s.players.forEach((p,i) => {
            s.moneySpentThisRound[i] = Math.max(0,get(`p${i}.spent`));
            p.money = Math.max(0,get(`p${i}.money`)); p.income = bounded(get(`p${i}.income`),-10,30); p.vp = Math.max(0,get(`p${i}.vp`));
            p.linksRemaining = {canal:bounded(get(`p${i}.canal`),0,14),rail:bounded(get(`p${i}.rail`),0,14)};
            types.forEach(t => p.industryTiles[t].forEach((tile,j) => {tile.used = j < get(`p${i}.used.${t}`);}));
            p.hand = cards.flatMap((c,j) => Array.from({length:bounded(get(`p${i}.card.${cardIds[j]}`),0,8)},()=>({...c}))).slice(0,8);
            p.hasWildLocation = p.hand.some(c=>c.type==='wildLocation'); p.hasWildIndustry = p.hand.some(c=>c.type==='wildIndustry');
        });
        s.boardIndustries = {}; s.breweryFarmTiles = {};
        slots.forEach(slot => {
            const owner = get(`slot.${slot}.owner`), type = types[get(`slot.${slot}.type`)-1];
            const data = D.INDUSTRY_DATA[type]?.find(t=>t.level===get(`slot.${slot}.level`));
            if(owner<1 || owner>s.numPlayers || !data) return;
            const tile = {playerId:owner-1,type,tileData:{...data,type},flipped:!!get(`slot.${slot}.flipped`),resourceCubes:bounded(get(`slot.${slot}.cubes`),0,8)};
            if(slot.startsWith('farm:')) s.breweryFarmTiles[slot.slice(5)]=tile; else s.boardIndustries[slot]=tile;
        });
        s.boardLinks = {};
        links.forEach(id => {const owner=get(`link.${id}.owner`); if(owner>0&&owner<=s.numPlayers) s.boardLinks[id]={playerId:owner-1,type:get(`link.${id}.rail`)?'rail':'canal'};});
        const merchantUsed = new Set();
        merchants.forEach((id,j) => {
            const i=s.merchantTiles.findIndex((m,i)=>!merchantUsed.has(i)&&`${m.location}:${m.buys||'*'}`===id);
            if(i<0)return;merchantUsed.add(i);
            s.merchantTiles[i].hasBeer=!!get(`merchant.${j}.${id}.beer`);
            s.merchantTiles[i].bonusClaimed=!!get(`merchant.${j}.${id}.claimed`);
        });
        return s;
    }
    function value(v, player) {
        const scores = Array.from({length:4},(_,p) => raw(v,`p${p}.vp`)+1.2*raw(v,`p${p}.income`)+0.06*raw(v,`p${p}.money`));
        slots.forEach(slot => {
            const owner = Math.round(raw(v,`slot.${slot}.owner`))-1;
            if(owner<0||owner>3) return;
            const flip=Math.max(0,Math.min(1,raw(v,`slot.${slot}.flipped`)));
            scores[owner] += Math.max(0,raw(v,`slot.${slot}.vp`))*(0.35+0.65*flip);
        });
        links.forEach(id => {const p=Math.round(raw(v,`link.${id}.owner`))-1;if(p>=0&&p<4)scores[p]+=1.5;});
        return scores[player]-0.25*Math.max(...scores.filter((_,p)=>p!==player));
    }
    const api = { schema, fields, indices, encodeState, encodeAction, decodeState, raw, value };
    if(node) module.exports=api; else root.BrassEncoding=api;
})(globalThis);
